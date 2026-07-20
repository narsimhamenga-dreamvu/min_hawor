import os
import cv2
import joblib
from tqdm import tqdm
import numpy as np
import torch
from hawor.utils.process import run_mano, run_mano_left

from lib.eval_utils.custom_utils import cam_to_img, load_gt_cam
from ultralytics import YOLO


if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def detect_track(imgfiles, thresh=0.5):
    
    hand_det_model = YOLO('./weights/external/detector.pt')

    # Run
    boxes_ = []
    tracks = {}
    for t, imgpath in enumerate(tqdm(imgfiles)):
        img_cv2 = cv2.imread(imgpath)

        ### --- Detection ---
        with torch.no_grad():
            with autocast():
                results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                boxes = np.hstack([boxes, confs[:, None]])
                find_right = False
                find_left = False
                for idx, box in enumerate(boxes):
                    if track_id[idx] == -1:
                        if handedness[[idx]] > 0:
                            id = int(10000)
                        else:
                            id = int(5000)
                    else:
                        id = track_id[idx]
                    subj = dict()
                    subj['frame'] = t 
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]
                    subj['det_handedness'] = handedness[[idx]]
                    
                    
                    if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]]==0):
                        if id in tracks:
                            tracks[id].append(subj)
                        else:
                            tracks[id] = [subj]

                        if handedness[[idx]] > 0:
                            find_right = True
                        elif handedness[[idx]] == 0:
                            find_left = True
    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, tracks


def detect_track_ego(imgfiles, thresh=0.2, sample_every=30, min_votes=2, molmo_batch=1):
    """YOLO hand detection/tracking filtered to the CAMERA-WEARER's hands using
    Molmo pointing (assist-YOLO, sampled + track-id propagation).

    1. YOLO .track() over all frames  -> per-frame boxes/handedness/track_ids
    2. Molmo (batched) on every `sample_every`-th frame -> ego-hand points
    3. each point votes for the track_id of the YOLO box it lands in;
       ego track_ids = those with >= `min_votes`
    4. emit tracks (same format as detect_track) keeping only ego track_ids,
       at most one left + one right (best conf) per frame.
    Molmo is loaded and freed inside this call so it doesn't hold VRAM during
    the later HaWoR/SLAM stages.
    """
    from collections import defaultdict
    from PIL import Image
    import gc
    from lib.pipeline.molmo_ego import load_molmo, point_ego_hands_batch

    hand_det_model = YOLO('./weights/external/detector.pt')

    # ---- Pass 1: YOLO tracking over all frames ----
    per_frame = []
    for imgpath in tqdm(imgfiles, desc='ego yolo-track'):
        img_cv2 = cv2.imread(imgpath)
        with torch.no_grad():
            with autocast():
                r = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)
        b = r[0].boxes
        n = len(b)
        per_frame.append((
            b.xyxy.cpu().numpy() if n else np.zeros((0, 4)),
            b.conf.cpu().numpy() if n else np.zeros((0,)),
            b.cls.cpu().numpy() if n else np.zeros((0,)),
            b.id.cpu().numpy() if b.id is not None else np.full(n, -1.0),
        ))

    # ---- Pass 2: Molmo pointing on sampled frames (batched) -> vote ego ids ----
    model, processor = load_molmo()
    sampled = list(range(0, len(imgfiles), sample_every))
    votes = defaultdict(int)
    for i in tqdm(range(0, len(sampled), molmo_batch), desc='ego molmo'):
        chunk = sampled[i:i + molmo_batch]
        imgs = [Image.open(imgfiles[s]).convert('RGB') for s in chunk]
        pts_list = point_ego_hands_batch(model, processor, imgs)
        for s, pts in zip(chunk, pts_list):
            boxes, conf, cls, ids = per_frame[s]
            for (px, py) in pts:
                for bi, tid in enumerate(ids):
                    if tid >= 0 and boxes[bi][0] <= px <= boxes[bi][2] and boxes[bi][1] <= py <= boxes[bi][3]:
                        votes[int(tid)] += 1
    ego_ids = {tid for tid, v in votes.items() if v >= min_votes}
    del model, processor
    gc.collect(); torch.cuda.empty_cache()

    # ---- Pass 3: build ego-only tracks (<=1 left + 1 right per frame) ----
    tracks = {}
    for t, (boxes, conf, cls, ids) in enumerate(per_frame):
        best = {}
        for bi, tid in enumerate(ids):
            if int(tid) not in ego_ids:
                continue
            side = 1 if cls[bi] > 0 else 0
            if side not in best or conf[bi] > best[side][0]:
                best[side] = (conf[bi], bi)
        for side, (c, bi) in best.items():
            subj = {'frame': t, 'det': True,
                    'det_box': np.hstack([boxes[bi], conf[bi]])[None, :],
                    'det_handedness': cls[[bi]]}
            tracks.setdefault(int(ids[bi]), []).append(subj)

    n_dets = sum(len(v) for v in tracks.values())
    print(f"[ego-filter] votes={dict(votes)} -> ego_ids={sorted(ego_ids)}; "
          f"kept {n_dets} dets, {len(tracks)} tracks (sampled {len(sampled)} frames, min_votes={min_votes})")
    return np.array([], dtype=object), np.array(tracks, dtype=object)


def parse_chunks(frame, boxes, min_len=16):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk)>=min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks

def parse_chunks_hand(frame, boxes, handedness, min_len=16):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    boxes_chunks = []
    handedness_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        handedness_chunk = handedness[start:bk]
        start = bk
        if len(f_chunk)>=min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)
            handedness_chunks.append(handedness_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            handedness_chunk = handedness[bk:]
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)
                handedness_chunks.append(handedness_chunk)

    return frame_chunks, boxes_chunks, handedness_chunks

def parse_chunks_hand_frame(frame):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        start = bk
        if len(f_chunk) > 0:
            frame_chunks.append(f_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            if len(f_chunk) > 0:
                frame_chunks.append(f_chunk)

    return frame_chunks

def detect_track_gt(imgfiles, out_path, thresh=0.5, min_size=None, 
                         device='cuda', save_vos=True, video_root=None, video=None, img_focal=None, K_matrix=None, test_video=False, 
                         dataset_type='hot3d', fix_shapedirs=True):
    # load GT mano
    gt_pth = os.path.join(os.path.dirname(out_path), 'anno.pth')
    datasets = joblib.load(gt_pth)
    world_rot = torch.stack([datasets['rot_l'], datasets['rot_r']])
    mano_valid = torch.any(world_rot != 0, dim=-1).numpy()
    world_trans = torch.stack([datasets['trans_l'], datasets['trans_r']])
    world_hand_pose = torch.stack([datasets['pose_l'], datasets['pose_r']])
    world_betas = torch.stack([datasets['betas_l'], datasets['betas_r']])

    # get joints
    gt_trans_l = world_trans[0:1]
    gt_rot_l = world_rot[0:1]
    gt_pose_l = world_hand_pose[0:1]
    gt_betas_l = world_betas[0:1]
    gt_trans_r = world_trans[1:2]
    gt_rot_r = world_rot[1:2]
    gt_pose_r = world_hand_pose[1:2]
    gt_betas_r = world_betas[1:2]

    target_glob_l = run_mano_left(gt_trans_l, gt_rot_l, gt_pose_l, betas=gt_betas_l, fix_shapedirs=fix_shapedirs)
    target_glob_r = run_mano(gt_trans_r, gt_rot_r, gt_pose_r, betas=gt_betas_r)
    world_joints = torch.stack((target_glob_l['joints'][0], target_glob_r['joints'][0]), dim=0).cpu() # B, T, 21, 3 

    R_w2c_gt, t_w2c_gt, _, _ = load_gt_cam(video_root, video, dataset_type=dataset_type)

    cam_j3d = torch.einsum("tij,btnj->btni", R_w2c_gt, world_joints) + t_w2c_gt[None, :, None, :]
    img_cv2 = cv2.imread(imgfiles[0])
    H, W, _ = img_cv2.shape
    img_center = [img_cv2.shape[0] / 2, img_cv2.shape[1] / 2]
    if not K_matrix is None:
        K = torch.from_numpy(K_matrix).float()
    else:
        K = torch.tensor(
            [
                [img_focal, 0, img_center[1]],
                [0, img_focal, img_center[0]],
                [0, 0, 1]
            ]
        )
    cam_j2d = cam_to_img(cam_j3d, K) # max value is H or W (B, T, 21, 2)
    x_coords = cam_j2d[..., 0]
    y_coords = cam_j2d[..., 1]
    
    valid_x = (x_coords >= 0) & (x_coords < W)
    valid_y = (y_coords >= 0) & (y_coords < H)
    valid = valid_x & valid_y
    valid_j2d = torch.sum(valid, axis=-1) >= 2 # (B,T)
    valid_j2d = valid_j2d & mano_valid

    # Run
    boxes_ = []
    handedness_ = []
    for t, imgpath in enumerate(tqdm(imgfiles)):
        with torch.no_grad():
            with autocast():
                # use GT
                boxes = []
                confs = []
                handedness = []
                if valid_j2d[0, t]: # has left hand
                    det_w = x_coords[0, t].max() - x_coords[0, t].min()
                    det_h = y_coords[0, t].max() - y_coords[0, t].min()
                    xmin = max(0, x_coords[0, t].min()-0.2*det_w)
                    ymin = max(0, y_coords[0, t].min()-0.2*det_h)
                    xmax = min(W, x_coords[0, t].max()+0.2*det_w)
                    ymax = min(H, y_coords[0, t].max()+0.2*det_h)
                    boxes.append([xmin, ymin, xmax, ymax])
                    confs.append(1)
                    handedness.append(0)
                if valid_j2d[1, t]: # has right hand
                    det_w = x_coords[1, t].max() - x_coords[1, t].min()
                    det_h = y_coords[1, t].max() - y_coords[1, t].min()
                    xmin = max(0, x_coords[1, t].min()-0.2*det_w)
                    ymin = max(0, y_coords[1, t].min()-0.2*det_h)
                    xmax = min(W, x_coords[1, t].max()+0.2*det_w)
                    ymax = min(H, y_coords[1, t].max()+0.2*det_h)
                    boxes.append([xmin, ymin, xmax, ymax])
                    confs.append(1)
                    handedness.append(1)
                if len(boxes):
                    boxes = np.array(boxes)
                    confs = np.array(confs)
                    handedness = np.array(handedness)
                else:
                    boxes = np.zeros((0,4))
                    confs = np.zeros((0,))
                    handedness = np.zeros((0,))
                boxes = np.hstack([boxes, confs[:, None]])

        boxes_.append(boxes)
        handedness_.append(handedness)

    ### --- Adapt tracks data structure ---
    tracks = {}
    for frame in range(len(boxes_)):         
        handedness = handedness_[frame]
        boxes = boxes_[frame]
        for idx in [0, 1]:  
            subj = {}
            sub_index = np.where(handedness == idx)[0][0] if np.any(handedness == idx) else None
            # add fields
            subj['frame'] = frame 
            if sub_index is None:
                subj['det'] = False
                subj['det_box'] = np.zeros([1, 5])
                subj['det_handedness'] = np.zeros([1,])
            else:
                subj['det'] = True
                subj['det_box'] = boxes[[sub_index]]
                subj['det_handedness'] = handedness[[sub_index]]
            
            if idx in tracks:
                tracks[idx].append(subj)
            else:
                tracks[idx] = [subj]

    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, tracks
