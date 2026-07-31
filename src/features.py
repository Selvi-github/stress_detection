import numpy as np

class FeatureFusion:
    def __init__(self):
        # Define baseline ranges for normalization (min, max)
        self.ranges = {
            'hr': (50, 120),
            'hrv': (0, 100),
            'ear': (0.1, 0.4),
            'mar': (0.0, 1.0),
            'blinks': (0, 30), # per minute approx
            'yawns': (0, 5),
            'pitch': (-45, 45),
            'yaw': (-45, 45),
            'roll': (-45, 45)
        }

    def _normalize(self, value, min_val, max_val):
        return np.clip((value - min_val) / (max_val - min_val + 1e-6), 0, 1)

    def fuse(self, hr, hrv, behavioral_metrics, emotion_probs):
        """
        Combines physiological, behavioral, and psychological data into a single feature vector.
        """
        features = []
        
        # 1. Physiological
        features.append(self._normalize(hr, *self.ranges['hr']))
        features.append(self._normalize(hrv, *self.ranges['hrv']))
        
        # 2. Behavioral
        if behavioral_metrics:
            features.append(self._normalize(behavioral_metrics['ear'], *self.ranges['ear']))
            features.append(self._normalize(behavioral_metrics['mar'], *self.ranges['mar']))
            features.append(self._normalize(behavioral_metrics['blinks'], *self.ranges['blinks']))
            features.append(self._normalize(behavioral_metrics['yawns'], *self.ranges['yawns']))
            features.append(self._normalize(behavioral_metrics['pitch'], *self.ranges['pitch']))
            features.append(self._normalize(behavioral_metrics['yaw'], *self.ranges['yaw']))
            features.append(self._normalize(behavioral_metrics['roll'], *self.ranges['roll']))
        else:
            features.extend([0] * 7)
            
        # 3. Psychological (Emotion Probabilities)
        emotions = ['Happy', 'Neutral', 'Sad', 'Fear', 'Angry', 'Disgust', 'Surprise']
        for e in emotions:
            features.append(emotion_probs.get(e, 0.0))
            
        return np.array(features)
        
    def get_feature_names(self):
        return [
            'HR_Norm', 'HRV_Norm', 
            'EAR_Norm', 'MAR_Norm', 'Blinks_Norm', 'Yawns_Norm', 
            'Pitch_Norm', 'Yaw_Norm', 'Roll_Norm',
            'Happy_Prob', 'Neutral_Prob', 'Sad_Prob', 'Fear_Prob', 
            'Angry_Prob', 'Disgust_Prob', 'Surprise_Prob'
        ]
