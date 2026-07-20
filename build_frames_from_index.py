"""Build extracted_images/ from a RealSense capture's color_frames/ PNGs, ordered
by frames_index.csv (frame_index column). Use this INSTEAD of ffmpeg-extracting
color.mp4 so the frames are the exact lossless captured frames in the canonical
order (and dropped frames are simply omitted, keeping a contiguous sequence).

PNG filename == frame_index (verified: color_frame_counter diverges and is NOT used).
Frames are symlinked as sequential 000000.jpg,000001.jpg,... (cv2/PIL decode by
content, so the .jpg name over a .png target is fine).

Usage: build_frames_from_index.py <src_session_dir> <out_extracted_images_dir>
"""
import csv, os, sys, glob

def main():
    src, out = sys.argv[1], sys.argv[2]
    cf = os.path.join(src, "color_frames")
    idx = os.path.join(src, "frames_index.csv")
    if not (os.path.isdir(cf) and os.path.isfile(idx)):
        print(f"NO_COLOR_FRAMES: {cf} or {idx} missing"); sys.exit(2)
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(open(idx)))
    rows.sort(key=lambda r: int(r["frame_index"]))
    seq = missing = 0
    for r in rows:
        fi = int(r["frame_index"])
        png = os.path.join(cf, f"{fi:06d}.png")
        if not os.path.exists(png):
            missing += 1; continue
        link = os.path.join(out, f"{seq:06d}.jpg")
        if not (os.path.islink(link) or os.path.exists(link)):
            os.symlink(png, link)
        seq += 1
    print(f"linked {seq} frames ({missing} dropped/missing of {len(rows)}) "
          f"from color_frames in frames_index order -> {out}")
    if seq == 0:
        sys.exit(3)

if __name__ == "__main__":
    main()
