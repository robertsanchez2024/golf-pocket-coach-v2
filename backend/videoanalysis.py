import cv2
import mediapipe as mp
from pose_landmarks import PoseLandmarkExtractor

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def find_address_frame(video_path: str, sample_every: int = 5) -> int:
    """
    Scans the video for the address/setup frame — the last still frame
    before the swing motion spike begins.

    Computes frame-to-frame pixel difference on sampled frames,
    then finds the last low-motion frame before the first big spike.
    Returns the frame index (in original video frame numbers).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return 0

    motion_scores = []  # (frame_index, score)
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(diff.mean())
                motion_scores.append((frame_idx, score))
            prev_gray = gray

        frame_idx += 1

    cap.release()

    if not motion_scores:
        return 0

    scores = [s for _, s in motion_scores]
    avg = sum(scores) / len(scores)
    threshold = avg * 1.5  # spike = 1.5x the average motion

    # Find the first big spike (swing starts)
    swing_start_idx = None
    for i, (fidx, score) in enumerate(motion_scores):
        if score > threshold:
            swing_start_idx = i
            break

    if swing_start_idx is None or swing_start_idx == 0:
        # No clear swing detected, use first frame
        return 0

    # Among all frames before the swing spike, pick the one with lowest motion
    pre_swing = motion_scores[:swing_start_idx]
    address_frame_idx, _ = min(pre_swing, key=lambda x: x[1])

    print(f"Address frame detected at frame {address_frame_idx}")
    return address_frame_idx


def extract_setup_frame(video_path: str, output_path: str = "setup_frame.png") -> int | None:
    """
    Finds the address frame and saves it as an image.
    Returns the frame index so it can be passed to pose analysis.
    """
    frame_number = find_address_frame(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read frame {frame_number}.")
        return None

    cv2.imwrite(output_path, frame)
    print(f"Setup frame saved as {output_path}")
    return frame_number


def skeleton_outline_onframe(image_path: str, output_path: str = "pose_output.png") -> bool:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No pose detected.")
            return False

        print("Pose detected — drawing skeleton.")
        output = image.copy()
        mp_drawing.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(output_path, output)
        print(f"Skeleton saved as {output_path}")
        return True


def print_stance_report(report: dict):
    for check, result in report.items():
        print(f"\n--- {check.upper().replace('_', ' ')} ---")
        for key, value in result.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze golf swing setup from a video file.")
    parser.add_argument("--video", default="swing.mp4", help="Video filename in the project folder")
    args = parser.parse_args()
    VIDEO = args.video

    frame_number = extract_setup_frame(VIDEO)
    if frame_number is None:
        print("Could not extract setup frame.")
        exit(1)

    skeleton_outline_onframe("setup_frame.png")

    print("\n=== STANCE ANALYSIS ===")
    extractor = PoseLandmarkExtractor(VIDEO)
    report = extractor.analyze_stance(frame_number=frame_number)
    print_stance_report(report)
