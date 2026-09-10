import cv2
import numpy as np
import os
import sys

def smootherstep(t):
    """
    Quintic polynomial smootherstep:
    6t^5 - 15t^4 + 10t^3
    First and second derivatives are zero at t=0 and t=1.
    Gives soft ease-in at start, linear cinematic push in middle, gentle ease-out at end.
    Zero sudden acceleration / jerk.
    """
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

def process_video_and_frames(
    input_path="proj%20x%20video%20(1)_1080p_202609091845_gwr_video_mvp.mp4",
    output_video_path="hero-video.mp4",
    max_zoom=1.32,  # 32% optical dolly-in (natural 35mm lens simulation)
    desktop_frames_count=120,
    mobile_frames_count=60,
    desktop_dir="frames/desktop",
    mobile_dir="frames/mobile"
):
    if not os.path.exists(input_path):
        # Fallback if filename encoding differs
        for cand in [
            "proj x video (1)_1080p_202609091845_gwr_video_mvp.mp4",
            "hero-video.mp4",
            "proj x video (1).mp4"
        ]:
            if os.path.exists(cand):
                input_path = cand
                break

    print(f"Reading source video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {input_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Source Specs: {w}x{h}, {total_frames} frames @ {fps} fps ({total_frames/fps:.2f}s duration)")

    os.makedirs(desktop_dir, exist_ok=True)
    os.makedirs(mobile_dir, exist_ok=True)

    # Prepare Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Calculate frame sampling indices for desktop and mobile
    desktop_indices = set(np.linspace(0, total_frames - 1, desktop_frames_count, dtype=int))
    mobile_indices = set(np.linspace(0, total_frames - 1, mobile_frames_count, dtype=int))

    desktop_frame_map = {idx: i for i, idx in enumerate(sorted(desktop_indices))}
    mobile_frame_map = {idx: i for i, idx in enumerate(sorted(mobile_indices))}

    print("Rendering cinematic optical push-in zoom...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Normalized timeline [0.0 -> 1.0]
        t = frame_idx / max(1, total_frames - 1)
        
        # Apply smootherstep ease
        ease_val = smootherstep(t)
        
        # Calculate continuous zoom factor
        zoom = 1.0 + (max_zoom - 1.0) * ease_val

        # Sub-pixel crop box centered on optical axis
        crop_w = w / zoom
        crop_h = h / zoom
        x1 = (w - crop_w) / 2.0
        y1 = (h - crop_h) / 2.0

        # Sub-pixel Affine mapping (prevents pixel jitter / aspect ratio warping)
        src_pts = np.float32([
            [x1, y1],
            [x1 + crop_w, y1],
            [x1, y1 + crop_h]
        ])
        dst_pts = np.float32([
            [0, 0],
            [w, 0],
            [0, h]
        ])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        
        # High fidelity Lanczos 4-tap sinc interpolation
        zoomed_frame = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Write to video
        out.write(zoomed_frame)

        # Export Desktop WebP frame if sampled
        if frame_idx in desktop_frame_map:
            d_out_idx = desktop_frame_map[frame_idx]
            d_path = os.path.join(desktop_dir, f"frame_{d_out_idx:04d}.webp")
            cv2.imwrite(d_path, zoomed_frame, [cv2.IMWRITE_WEBP_QUALITY, 92])

        # Export Mobile WebP frame if sampled (scaled to mobile resolution, e.g., 960x540)
        if frame_idx in mobile_frame_map:
            m_out_idx = mobile_frame_map[frame_idx]
            m_path = os.path.join(mobile_dir, f"frame_{m_out_idx:04d}.webp")
            mobile_frame = cv2.resize(zoomed_frame, (960, 540), interpolation=cv2.INTER_AREA)
            cv2.imwrite(m_path, mobile_frame, [cv2.IMWRITE_WEBP_QUALITY, 85])

        if (frame_idx + 1) % 40 == 0 or frame_idx == total_frames - 1:
            print(f"Processed frame {frame_idx + 1}/{total_frames} (zoom: {zoom:.3f}x)")

    out.release()
    cap.release()

    # Also make a copy for the original long filename
    try:
        import shutil
        shutil.copyfile(output_video_path, "proj_x_video_cinematic_zoom_1080p.mp4")
    except Exception as e:
        print(f"Warning: {e}")

    print(f"\nCompleted successfully!")
    print(f"Output Video: {output_video_path} ({os.path.getsize(output_video_path)/1024/1024:.2f} MB)")
    print(f"Desktop WebP frames: {desktop_frames_count} frames saved in {desktop_dir}/")
    print(f"Mobile WebP frames: {mobile_frames_count} frames saved in {mobile_dir}/")

if __name__ == "__main__":
    process_video_and_frames()
