import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

class EmotionAnalyzer:
    def __init__(self, model_path="models/emotion_rf.pkl"):
        self.emotions = ['Happy', 'Neutral', 'Sad', 'Fear', 'Angry', 'Disgust', 'Surprise']
        self.model_path = model_path
        self.model = None
        
        # Load model if it exists, otherwise use a fallback or mock
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            
    def _extract_features(self, landmarks):
        """
        Extract pairwise distances or AU-like features from landmarks.
        """
        if not landmarks:
            return None
            
        # Select key landmarks for emotion (eyes, eyebrows, mouth, jaw)
        key_indices = [
            33, 133, 362, 263, # Eyes outer/inner
            70, 300,           # Eyebrows
            61, 291, 0, 17,    # Mouth outer
            13, 14,            # Mouth inner
            152                # Chin
        ]
        
        pts = [landmarks[i] for i in key_indices]
        features = []
        
        # Compute normalized pairwise distances
        chin = pts[-1]
        for i in range(len(pts)-1):
            for j in range(i+1, len(pts)-1):
                # Normalize by distance to chin to be scale invariant
                d = np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))
                norm_factor = np.linalg.norm(np.array(pts[0]) - np.array(chin)) + 1e-6
                features.append(d / norm_factor)
                
        return np.array(features).reshape(1, -1)
        
    def analyze(self, landmarks):
        features = self._extract_features(landmarks)
        if features is None:
            return "Neutral", {e: 0.0 for e in self.emotions}
            
        if self.model is not None:
            preds = self.model.predict_proba(features)[0]
            pred_idx = np.argmax(preds)
            return self.emotions[pred_idx], {self.emotions[i]: preds[i] for i in range(len(self.emotions))}
        else:
            # Fallback mock logic if model isn't trained yet
            # E.g., if mouth is very open -> Surprise
            # If corners of mouth are up -> Happy
            mouth_width = np.linalg.norm(np.array(landmarks[61]) - np.array(landmarks[291]))
            mouth_height = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[17]))
            
            ratio = mouth_height / (mouth_width + 1e-6)
            if ratio > 0.6:
                emotion = "Surprise"
            elif ratio < 0.2:
                emotion = "Neutral"
            else:
                emotion = "Happy" # Oversimplified, just for mock fallback
                
            probs = {e: 0.1 for e in self.emotions}
            probs[emotion] = 0.4
            
            return emotion, probs
