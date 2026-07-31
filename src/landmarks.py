import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define ROI indices based on MediaPipe Face Mesh
        # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
        self.ROI_INDICES = {
            'forehead': [103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 368, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70],
            'left_cheek': [118, 119, 100, 126, 209, 49, 50, 205, 206, 207, 214, 212, 210, 169], # Left from viewer's perspective
            'right_cheek': [347, 348, 329, 355, 429, 279, 280, 425, 426, 427, 434, 432, 430, 394],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
            'nose': [8, 411, 327, 326, 2, 97, 98, 187, 19],
            'jaw': [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 377, 152]
        }

    def process(self, image, face_bbox=None):
        """
        Extract landmarks and ROIs.
        Args:
            image: BGR image
            face_bbox: Optional bounding box (x, y, w, h) to crop and speed up mesh or filter results.
        Returns:
            landmarks_coords: list of (x, y) tuples for 468 points
            rois: dict of masks for each region
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.face_mesh.process(rgb_image)
        rgb_image.flags.writeable = True

        h, w, _ = image.shape
        all_landmarks = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                coords = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
                
                # If bbox is provided, check if this mesh belongs to the bbox
                if face_bbox is not None:
                    bx, by, bw, bh = face_bbox
                    # Very simple check: is the nose tip inside the bbox?
                    nose_x, nose_y = coords[1]
                    if not (bx <= nose_x <= bx + bw and by <= nose_y <= by + bh):
                        continue
                
                all_landmarks = coords
                break  # process only one face for now that matches

        if not all_landmarks:
            return None, None

        # Generate Masks for ROIs
        masks = {}
        for region_name, indices in self.ROI_INDICES.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            region_points = np.array([all_landmarks[i] for i in indices], dtype=np.int32)
            # Fill convex hull of the points to create a solid mask
            hull = cv2.convexHull(region_points)
            cv2.fillConvexPoly(mask, hull, 255)
            masks[region_name] = mask

        return all_landmarks, masks 

    def get_roi_pixels(self, image, mask):
        """
        Extract the pixels from the image given a mask.
        Returns a 1D array of pixels or the mean color.
        """
        # Apply mask
        masked_img = cv2.bitwise_and(image, image, mask=mask)
        # Get pixels where mask is 255
        pixels = masked_img[mask == 255]
        return pixels

    def get_roi_mean_color(self, image, mask):
        """
        Returns (B, G, R) mean color of the ROI.
        """
        pixels = self.get_roi_pixels(image, mask)
        if len(pixels) == 0:
            return (0, 0, 0)
        return np.mean(pixels, axis=0)

