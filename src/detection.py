import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict() # {object_id: centroid}
        self.disappeared = OrderedDict() # {object_id: frames_disappeared}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # rects: list of (startX, startY, endX, endY)
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find the smallest value in each row and then sort the row indexes based on their minimum values
            rows = D.min(axis=1).argsort()

            # Find the smallest value in each column and then sort using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=100)

    def process(self, image):
        """
        Detect faces in an image and track them.
        Args:
            image: BGR image from OpenCV
        Returns:
            tracked_faces: Dictionary {object_id: {'bbox': (x, y, w, h), 'confidence': conf}}
        """
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference
        rgb_image.flags.writeable = False
        results = self.detector.process(rgb_image)
        rgb_image.flags.writeable = True

        h, w, _ = image.shape
        rects = []
        confidences = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                xmin = int(bboxC.xmin * w)
                ymin = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Ensure bounding box is within image dimensions
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmin + width)
                ymax = min(h, ymin + height)

                if xmax > xmin and ymax > ymin:
                    rects.append((xmin, ymin, xmax, ymax))
                    confidences.append(detection.score[0])

        # Update tracker
        objects = self.tracker.update(rects)

        # Match objects with bounding boxes
        tracked_faces = {}
        for object_id, centroid in objects.items():
            # Find the bounding box closest to the centroid
            min_dist = float("inf")
            best_rect = None
            best_conf = 0.0
            
            for i, rect in enumerate(rects):
                startX, startY, endX, endY = rect
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                d = dist.euclidean(centroid, (cX, cY))
                if d < min_dist:
                    min_dist = d
                    best_rect = rect
                    best_conf = confidences[i]
            
            if best_rect is not None and min_dist < 50:
                x, y, xmax, ymax = best_rect
                tracked_faces[object_id] = {
                    'bbox': (x, y, xmax - x, ymax - y),
                    'confidence': best_conf
                }

        return tracked_faces
