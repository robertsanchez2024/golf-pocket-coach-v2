from pose_landmarks import PoseLandmarkExtractor
from videoanalysis import find_address_frame

def analyze_video(video_path: str) -> dict:
    
    frame_number = find_address_frame(video_path)

    extractor = PoseLandmarkExtractor(video_path)
    report = extractor.analyze_stance(frame_number=frame_number)

    return report