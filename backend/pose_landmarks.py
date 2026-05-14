import cv2
import mediapipe as mp
import numpy as np

LANDMARK_NAMES = {
    0:  "Nose",
    1:  "Left Eye Inner",
    2:  "Left Eye",
    3:  "Left Eye Outer",
    4:  "Right Eye Inner",
    5:  "Right Eye",
    6:  "Right Eye Outer",
    7:  "Left Ear",
    8:  "Right Ear",
    9:  "Mouth Left",
    10: "Mouth Right",
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist",
    17: "Left Pinky",
    18: "Right Pinky",
    19: "Left Index Finger",
    20: "Right Index Finger",
    21: "Left Thumb",
    22: "Right Thumb",
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle",
    29: "Left Heel",
    30: "Right Heel",
    31: "Left Foot",
    32: "Right Foot",
}

# Pro reference ranges (degrees) — based on PGA tour biomechanics data
# Sources: TPI (Titleist Performance Institute), PGA teaching standards
KNEE_FLEX_RANGE = (15, 25)   # Rory/Tiger both ~15-20° at address
SPINE_TILT_RANGE = (3, 10)   # Trail shoulder lower; lateral tilt at address
HIP_HINGE_RANGE = (30, 45)   # Forward bend from vertical; best from down-the-line view


class GolferLandmarks:
    def __init__(self, landmarks):
        lm = landmarks
        P = mp.solutions.pose.PoseLandmark

        self.left_shoulder  = lm[P.LEFT_SHOULDER]
        self.right_shoulder = lm[P.RIGHT_SHOULDER]
        self.left_hip       = lm[P.LEFT_HIP]
        self.right_hip      = lm[P.RIGHT_HIP]
        self.left_knee      = lm[P.LEFT_KNEE]
        self.right_knee     = lm[P.RIGHT_KNEE]
        self.left_ankle     = lm[P.LEFT_ANKLE]
        self.right_ankle    = lm[P.RIGHT_ANKLE]
        self.left_elbow     = lm[P.LEFT_ELBOW]
        self.right_elbow    = lm[P.RIGHT_ELBOW]
        self.left_wrist     = lm[P.LEFT_WRIST]
        self.right_wrist    = lm[P.RIGHT_WRIST]


class PoseLandmarkExtractor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._mp_pose = mp.solutions.pose
        self._landmarks = None

    def extract_landmarks(self, frame_number: int = 0) -> dict | None:
        """
        Extract pose landmarks from a specific frame in the video.
        Returns a dict mapping real body part names to (x, y, z, visibility).
        x and y are normalized [0, 1]; z is depth relative to hips.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Could not open video: {self.video_path}")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Could not read frame {frame_number}.")
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self._mp_pose.Pose(static_image_mode=True) as pose:
            results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            print("No pose detected in frame.")
            return None

        landmarks = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            name = LANDMARK_NAMES.get(idx, f"Landmark {idx}")
            landmarks[name] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }

        self._landmarks = landmarks
        return landmarks

    def _get_point(self, name: str) -> np.ndarray | None:
        if not self._landmarks or name not in self._landmarks:
            return None
        lm = self._landmarks[name]
        return np.array([lm["x"], lm["y"]])

    def _get_point_3d(self, name: str) -> np.ndarray | None:
        """Returns (x, y, z) — z is depth relative to hips, same scale as x."""
        if not self._landmarks or name not in self._landmarks:
            return None
        lm = self._landmarks[name]
        return np.array([lm["x"], lm["y"], lm["z"]])

    def get_joint_angle(self, a: str, joint: str, b: str) -> float | None:
        """
        Angle in degrees at `joint` between the vectors joint->a and joint->b.
        Uses 3D coordinates to compensate for camera perspective.
        Straight line = 180°, fully bent = 0°.
        """
        pa = self._get_point_3d(a)
        pj = self._get_point_3d(joint)
        pb = self._get_point_3d(b)

        if any(p is None for p in [pa, pj, pb]):
            return None

        v1 = pa - pj
        v2 = pb - pj
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    def check_knee_flex(self) -> dict:
        """
        Knee flex at address. Measured as 180° minus the hip-knee-ankle angle.
        PGA pro standard (Rory, Tiger): 15-25° of flex.
        """
        left_angle = self.get_joint_angle("Left Hip", "Left Knee", "Left Ankle")
        right_angle = self.get_joint_angle("Right Hip", "Right Knee", "Right Ankle")

        left_flex = round(180 - left_angle, 1) if left_angle is not None else None
        right_flex = round(180 - right_angle, 1) if right_angle is not None else None

        def evaluate(flex: float | None) -> str:
            if flex is None:
                return "undetected"
            lo, hi = KNEE_FLEX_RANGE
            if flex < lo:
                return "too straight — flex your knees more"
            if flex > hi:
                return "too bent — stand a bit taller"
            return "good"

        return {
            "left_knee_flex_deg":    left_flex,
            "left_knee_status":      evaluate(left_flex),
            "right_knee_flex_deg":   right_flex,
            "right_knee_status":     evaluate(right_flex),
            "pro_range_deg":         KNEE_FLEX_RANGE,
        }

    def check_spine_tilt(self) -> dict:
        """
        Lateral spine tilt at address.
        The trail shoulder should sit slightly lower than the lead shoulder.
        PGA pro standard: 3-10° tilt (right shoulder lower for right-handed golfers).

        Uses 3D coords: tilt = arctan(vertical_diff / 3D_horizontal_dist)
        so the depth component cancels out perspective distortion from off-axis cameras.
        """
        ls = self._get_point_3d("Left Shoulder")
        rs = self._get_point_3d("Right Shoulder")

        if ls is None or rs is None:
            return {"status": "undetected"}

        # y increases downward in image coords
        dy = rs[1] - ls[1]
        # 3D horizontal separation (x + depth) removes perspective foreshortening
        horiz_dist = np.sqrt((rs[0] - ls[0])**2 + (rs[2] - ls[2])**2)

        tilt_deg = round(float(np.degrees(np.arctan2(abs(dy), horiz_dist))), 1)
        lower_shoulder = "right" if dy > 0 else "left"

        lo, hi = SPINE_TILT_RANGE
        if tilt_deg < lo:
            status = "shoulders too level — let trail shoulder drop slightly"
        elif tilt_deg > hi:
            status = "too much tilt — shoulders are too uneven"
        else:
            status = "good"

        return {
            "tilt_degrees":    tilt_deg,
            "lower_shoulder":  lower_shoulder,
            "status":          status,
            "pro_range_deg":   SPINE_TILT_RANGE,
        }

    def check_hip_hinge(self) -> dict:
        """
        Forward bend / hip hinge. Measures the angle of the torso from vertical
        in the sagittal plane (using the z/depth component of the torso vector).

        By projecting onto the y-z plane (height + depth), this works from any
        camera angle — not just down-the-line.
        PGA pro standard: 30-45° of forward bend.
        """
        lh = self._get_point_3d("Left Hip")
        rh = self._get_point_3d("Right Hip")
        ls = self._get_point_3d("Left Shoulder")
        rs = self._get_point_3d("Right Shoulder")

        if any(p is None for p in [lh, rh, ls, rs]):
            return {"status": "undetected"}

        # Use 2D midpoints — hip hinge is measured in the image plane.
        # For best results film from a down-the-line angle (camera behind the golfer).
        lh2 = self._get_point("Left Hip")
        rh2 = self._get_point("Right Hip")
        ls2 = self._get_point("Left Shoulder")
        rs2 = self._get_point("Right Shoulder")

        mid_hip = (lh2 + rh2) / 2
        mid_shoulder = (ls2 + rs2) / 2
        torso_vec = mid_shoulder - mid_hip

        vertical = np.array([0.0, -1.0])  # upward in image y-coords
        cos_angle = np.dot(torso_vec, vertical) / (
            np.linalg.norm(torso_vec) * np.linalg.norm(vertical)
        )
        forward_bend = round(float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))), 1)

        lo, hi = HIP_HINGE_RANGE
        if forward_bend < lo:
            status = "standing too upright — hinge more at the hips"
        elif forward_bend > hi:
            status = "bending too much — stand slightly taller"
        else:
            status = "good"

        return {
            "forward_bend_deg": forward_bend,
            "status":           status,
            "pro_range_deg":    HIP_HINGE_RANGE,
            "note":             "Film from a down-the-line angle for best accuracy.",
        }

    def analyze_stance(self, frame_number: int = 0) -> dict:
        """
        Run all stance checks on a single frame and return a full report.
        Call this instead of extract_landmarks() + individual checks manually.
        """
        landmarks = self.extract_landmarks(frame_number)
        if landmarks is None:
            return {"error": "No pose detected — check video path and frame number."}

        return {
            "knee_flex":   self.check_knee_flex(),
            "spine_tilt":  self.check_spine_tilt(),
            "hip_hinge":   self.check_hip_hinge(),
        }


if __name__ == "__main__":
    extractor = PoseLandmarkExtractor("swing.mp4")
    report = extractor.analyze_stance(frame_number=0)

    for check, result in report.items():
        print(f"\n--- {check.upper().replace('_', ' ')} ---")
        for key, value in result.items():
            print(f"  {key}: {value}")
