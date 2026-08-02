import cv2
import threading
import time
import sys
import numpy as np

from src.detection import FaceDetector
from src.landmarks import LandmarkExtractor
from src.rppg import RPPGEngine
from src.behavioral import BehavioralFeaturesExtractor
from src.emotion import EmotionAnalyzer
from src.features import FeatureFusion
from src.stress_classifier import StressClassifier
from src.database import DatabaseManager
from src.report_generator import ReportGenerator
from src.ui import StressDashboard

class StressDetectionApp:
    def __init__(self):
        # Initialize Core Vision
        self.detector = FaceDetector()
        self.landmarks_extractor = LandmarkExtractor()
        
        # Initialize Engines
        self.rppg = RPPGEngine(fps=30, window_size_sec=10)
        self.behavioral = BehavioralFeaturesExtractor(fps=30)
        self.emotion_analyzer = EmotionAnalyzer()
        
        # Initialize ML & Fusion
        self.fusion = FeatureFusion()
        self.classifier = StressClassifier()
        
        # Initialize DB & Reports
        self.db = DatabaseManager()
        self.session_id = self.db.start_session("Employee_Current")
        self.report_gen = ReportGenerator()
        
        # Initialize UI
        self.ui = StressDashboard(
            on_closing_callback=self.stop,
            end_session_callback=self.end_session,
            upload_img_callback=self.set_image_source,
            upload_vid_callback=self.set_video_source,
            webcam_callback=self.set_webcam_source,
            dual_phone_callback=self.set_dual_phone_source
        )
        
        # Source State
        self.source_type = "webcam" # "webcam", "video", "image", or "dual_phone"
        self.static_image = None
        self.cap = None
        self.cap1 = None
        self.cap2 = None
        self.phone_url1 = "http://192.168.1.4:8080/video"
        self.phone_url2 = "http://192.168.1.3:8080/video"
        
        self.open_webcam_capture()
        
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_loop)
        
        # To store latest metrics for UI
        self.latest_frame = None
        self.latest_signal = []
        self.latest_metrics = {
            'hr': 0.0, 'hrv': 0.0, 'blinks': 0, 'yawns': 0, 
            'emotion': 'Neutral', 'stress_score': 0.0, 'stress_level': 'Normal'
        }
        self.latest_feature_vector = None
        self.feature_names = self.fusion.get_feature_names()
        
        # DB Logging Control
        self.last_db_log = time.time()
        self.db_log_interval = 2.0 # Log every 2 seconds

    def release_all_captures(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.cap1 is not None:
            self.cap1.release()
            self.cap1 = None
        if self.cap2 is not None:
            self.cap2.release()
            self.cap2 = None

    def open_webcam_capture(self):
        self.release_all_captures()
            
        # Try DirectShow backend first on Windows for instant capture, fallback to default
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            self.cap = cap
            return True
        return False

    def set_image_source(self, image_path):
        img = cv2.imread(image_path)
        if img is not None:
            self.release_all_captures()
            self.static_image = img
            self.source_type = "image"
            self.classifier.clear_history()
            print(f"Switched source to Image: {image_path}")

    def set_video_source(self, video_path):
        self.release_all_captures()
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.cap = cap
            self.source_type = "video"
            self.classifier.clear_history()
            print(f"Switched source to Video: {video_path}")

    def set_webcam_source(self):
        if self.open_webcam_capture():
            self.source_type = "webcam"
            self.classifier.clear_history()
            print("Switched source to Webcam")

    def set_dual_phone_source(self):
        self.release_all_captures()
        print(f"Connecting to Phone 1 ({self.phone_url1}) and Phone 2 ({self.phone_url2})...")
        cap1 = cv2.VideoCapture(self.phone_url1)
        cap2 = cv2.VideoCapture(self.phone_url2)
        
        if cap1.isOpened() and cap2.isOpened():
            self.cap1 = cap1
            self.cap2 = cap2
            self.source_type = "dual_phone"
            self.classifier.clear_history()
            print("SUCCESS: Dual Phone Cameras Connected!")
        else:
            print("Error connecting to one or both phone cameras. Retrying single fallback...")
            if cap1.isOpened():
                self.cap = cap1
                self.source_type = "video"
            elif cap2.isOpened():
                self.cap = cap2
                self.source_type = "video"

    def process_single_frame(self, frame, cam_label="Camera"):
        display_frame = frame.copy()
        cv2.putText(display_frame, f"📷 {cam_label}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 1. Face Detection & Tracking
        tracked_faces = self.detector.process(frame)
        
        if tracked_faces:
            object_id, data = list(tracked_faces.items())[0]
            x, y, w, h = data['bbox']
            
            # Draw BBox
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"ID: {object_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 2. Landmarks & ROI
            landmarks, masks = self.landmarks_extractor.process(frame, face_bbox=(x, y, w, h))
            
            if landmarks and masks:
                # Draw landmarks
                for pt in landmarks:
                    cv2.circle(display_frame, pt, 1, (0, 0, 255), -1)
                
                # 3. rPPG (Only for Video Streams / Webcam, not static single photos)
                if self.source_type != "image":
                    mask_comb = cv2.bitwise_or(masks['forehead'], masks['left_cheek'])
                    mask_comb = cv2.bitwise_or(mask_comb, masks['right_cheek'])
                    rgb_mean = self.landmarks_extractor.get_roi_mean_color(frame, mask_comb)
                    
                    if np.any(rgb_mean):
                        self.rppg.add_frame_mean(rgb_mean)
                        
                    hr, hrv, filtered_sig = self.rppg.estimate_heart_rate(method='POS')
                    self.latest_signal = filtered_sig
                else:
                    hr, hrv = 0.0, 0.0
                    self.latest_signal = []
                
                # 4. Behavioral
                behav_metrics = self.behavioral.update(landmarks, frame.shape)
                
                # 5. Emotion
                emotion, emotion_probs = self.emotion_analyzer.analyze(landmarks)
                
                # 6. Feature Fusion
                feature_vector = self.fusion.fuse(hr, hrv, behav_metrics, emotion_probs)
                self.latest_feature_vector = feature_vector
                
                # 7. Stress Classification
                score, level, feature_importance = self.classifier.predict(feature_vector, self.feature_names)
                
                # Overlay results on frame
                level_color = (0, 255, 0) if level == "Normal" else ((0, 165, 255) if level == "Acute Stress" else (0, 0, 255))
                cv2.putText(display_frame, f"Stress: {score:.1f} ({level})", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 2)
                
                # Update metrics store
                self.latest_metrics = {
                    'hr': hr, 'hrv': hrv, 
                    'blinks': behav_metrics['blinks'] if behav_metrics else 0,
                    'yawns': behav_metrics['yawns'] if behav_metrics else 0,
                    'emotion': emotion, 'stress_score': score, 'stress_level': level
                }
                
                # Database Logging
                if time.time() - self.last_db_log > self.db_log_interval:
                    self.db.log_metrics(
                        self.session_id, hr, hrv, 
                        self.latest_metrics['blinks'], self.latest_metrics['yawns'],
                        emotion, score, level
                    )
                    self.last_db_log = time.time()
        else:
            mean_b = np.mean(display_frame)
            if mean_b < 2.0 and self.source_type == "webcam":
                cv2.putText(display_frame, "Camera Lens Blocked / Shutter Closed!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "No Face Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        return display_frame

    def process_loop(self):
        while self.running:
            if self.source_type == "dual_phone":
                if self.cap1 is None or self.cap2 is None or not self.cap1.isOpened() or not self.cap2.isOpened():
                    time.sleep(0.05)
                    continue
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                if not ret1 or not ret2 or frame1 is None or frame2 is None:
                    time.sleep(0.02)
                    continue
                
                frame1 = cv2.resize(frame1, (640, 480))
                frame2 = cv2.resize(frame2, (640, 480))
                
                f1_proc = self.process_single_frame(frame1, cam_label="Phone 1 (192.168.1.4)")
                f2_proc = self.process_single_frame(frame2, cam_label="Phone 2 (192.168.1.3)")
                
                display_frame = np.hstack([f1_proc, f2_proc])
            elif self.source_type == "image":
                if self.static_image is None:
                    time.sleep(0.05)
                    continue
                frame = cv2.resize(self.static_image.copy(), (640, 480))
                display_frame = self.process_single_frame(frame, cam_label="Uploaded Image")
            else:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.05)
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    if self.source_type == "video":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                        
                frame = cv2.resize(frame, (640, 480))
                display_frame = self.process_single_frame(frame, cam_label="Webcam / Video Feed")

            self.latest_frame = display_frame
            time.sleep(0.01)

    def ui_update_loop(self):
        """
        Periodically updates the Tkinter UI from the processing thread data.
        """
        if self.latest_frame is not None:
            self.ui.update_video(self.latest_frame)
            
        self.ui.update_metrics(**self.latest_metrics)
        self.ui.update_plot(self.latest_signal)
        
        # Schedule next update
        if self.running:
            self.ui.after(30, self.ui_update_loop)

    def start(self):
        self.processing_thread.start()
        self.ui.after(30, self.ui_update_loop)
        self.ui.mainloop()

    def end_session(self):
        print("Ending session and generating report...")
        self.db.end_session(self.session_id)
        
        # Generate SHAP Plot for the last recorded feature vector
        shap_plot_path = None
        if self.latest_feature_vector is not None:
            shap_plot_path = self.classifier.generate_shap_plot(self.latest_feature_vector, self.feature_names)
            
        # Get metrics DataFrame
        df = self.db.get_session_data(self.session_id)
        
        # Generate Report
        report_path = self.report_gen.generate_report(self.session_id, df, shap_plot_path)
        print(f"Report generated: {report_path}")
        
        self.stop()

    def stop(self):
        self.running = False
        self.cap.release()
        self.db.close()
        self.ui.destroy()
        sys.exit(0)

if __name__ == "__main__":
    app = StressDetectionApp()
    app.start()
