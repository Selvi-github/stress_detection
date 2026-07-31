import numpy as np
import cv2
from scipy.spatial import distance as dist
import time
from collections import deque

class BehavioralFeaturesExtractor:
    def __init__(self, fps=30):
        self.fps = fps
        self.blink_counter = 0
        self.yawn_counter = 0
        
        self.ear_history = deque(maxlen=fps * 10) # 10 seconds
        self.mar_history = deque(maxlen=fps * 10)
        
        self.EAR_THRESHOLD = 0.20
        self.MAR_THRESHOLD = 0.50
        
        self.eye_closed_frames = 0
        self.mouth_open_frames = 0

    def compute_aspect_ratio(self, points):
        """
        Compute Aspect Ratio given 6 points (like eye or mouth).
        points should be a list/array of (x, y) coordinates.
        Usually arranged as:
          1   2
        0       3
          5   4
        """
        if len(points) < 6:
            return 0.0
            
        A = dist.euclidean(points[1], points[5])
        B = dist.euclidean(points[2], points[4])
        C = dist.euclidean(points[0], points[3])
        
        if C == 0:
            return 0.0
        
        ar = (A + B) / (2.0 * C)
        return ar

    def get_ear(self, landmarks):
        
        # Sub-sample 6 points roughly matching the traditional EAR formula
        # MediaPipe has more points, we pick the top/bottom/left/right extremes
        # Left eye: 33 (L), 133 (R), 159 (T1), 158 (T2), 145 (B1), 153 (B2)
        # Right eye: 362 (L), 263 (R), 386 (T1), 385 (T2), 374 (B1), 373 (B2)
        
        # We will use simplified indexing for EAR calculation
        le_pts = [landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]]
        re_pts = [landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]]
        
        left_ear = self.compute_aspect_ratio(le_pts)
        right_ear = self.compute_aspect_ratio(re_pts)
        
        return (left_ear + right_ear) / 2.0

    def get_mar(self, landmarks):
        # Outer lip landmarks roughly: 61 (L), 291 (R), 37 (T1), 267 (T2), 84 (B1), 314 (B2)
        mouth_pts = [landmarks[61], landmarks[37], landmarks[267], landmarks[291], landmarks[314], landmarks[84]]
        return self.compute_aspect_ratio(mouth_pts)

    def estimate_head_pose(self, landmarks, image_shape):
        """
        Estimate Head Pose (Pitch, Yaw, Roll) using solvePnP
        """
        h, w, _ = image_shape
        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points from MediaPipe
        image_points = np.array([
            landmarks[1],     # Nose tip
            landmarks[152],   # Chin
            landmarks[33],    # Left eye left corner
            landmarks[263],   # Right eye right corner
            landmarks[61],    # Left mouth corner
            landmarks[291]    # Right mouth corner
        ], dtype="double")
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0, 0, 0
            
        # Get euler angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        
        pitch = euler_angles[0, 0]
        yaw = euler_angles[1, 0]
        roll = euler_angles[2, 0]
        
        return pitch, yaw, roll

    def update(self, landmarks, image_shape):
        """
        Update state with new landmarks and return behavioral metrics.
        """
        if not landmarks:
            return None
            
        ear = self.get_ear(landmarks) # Indices hardcoded inside
        mar = self.get_mar(landmarks)
        pitch, yaw, roll = self.estimate_head_pose(landmarks, image_shape)
        
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        
        # Blink detection
        if ear < self.EAR_THRESHOLD:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= 2: # At least 2 frames closed to count as blink
                self.blink_counter += 1
            self.eye_closed_frames = 0
            
        # Yawn detection
        if mar > self.MAR_THRESHOLD:
            self.mouth_open_frames += 1
        else:
            if self.mouth_open_frames >= 15: # Mouth open for a while
                self.yawn_counter += 1
            self.mouth_open_frames = 0
            
        return {
            'ear': ear,
            'mar': mar,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'blinks': self.blink_counter,
            'yawns': self.yawn_counter
        }
