import numpy as np
import os
import joblib
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class StressClassifier:
    def __init__(self, model_path="models/stress_rf.pkl"):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        
        # We use a Regressor to output a continuous Stress Score (0-100)
        # If the model doesn't exist, we'll initialize a mock one for the demo
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self._train_mock_model()

    def _train_mock_model(self):
        """
        Train a dummy model so the system runs out-of-the-box.
        In a real scenario, this is trained on WESAD or UBFC datasets.
        """
        # Feature order matches features.py:
        # HR, HRV, EAR, MAR, Blinks, Yawns, Pitch, Yaw, Roll, Happy, Neutral, Sad, Fear, Angry, Disgust, Surprise
        X_dummy = np.random.rand(100, 16)
        
        # Mock logic: High HR, High Yawns, High Angry/Fear -> High Stress
        y_dummy = (
            X_dummy[:, 0] * 30 +   # HR
            (1 - X_dummy[:, 1]) * 20 + # Low HRV -> High Stress
            X_dummy[:, 5] * 10 +   # Yawns
            X_dummy[:, 13] * 20 +  # Angry
            X_dummy[:, 12] * 20    # Fear
        )
        y_dummy = np.clip(y_dummy, 0, 100)
        
        self.model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.model.fit(X_dummy, y_dummy)
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, feature_vector, feature_names):
        """
        Predict stress score and return SHAP explanation.
        """
        if feature_vector is None or len(feature_vector) == 0:
            return 0.0, "Low", None

        # Predict score (0 - 100)
        score = self.model.predict(feature_vector.reshape(1, -1))[0]
        score = np.clip(score, 0, 100)

        # Categorize
        if score < 33:
            level = "Low"
        elif score < 66:
            level = "Moderate"
        else:
            level = "High"

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(feature_vector.reshape(1, -1))
        
        # Extract top contributing features (positive and negative)
        # shap_values[0] because it's a single sample
        contributions = shap_values[0] if isinstance(shap_values, list) else shap_values
        if len(contributions.shape) > 1:
            contributions = contributions[0]
            
        feature_importance = []
        for i, val in enumerate(contributions):
            feature_importance.append({
                'feature': feature_names[i],
                'value': float(feature_vector[i]),
                'contribution': float(val)
            })
            
        # Sort by absolute contribution
        feature_importance.sort(key=lambda x: abs(x['contribution']), reverse=True)

        return score, level, feature_importance

    def generate_shap_plot(self, feature_vector, feature_names, save_path="reports/shap_plot.png"):
        """
        Generates and saves a SHAP waterfall or force plot for the current prediction.
        """
        if self.explainer is None:
            return None
            
        shap_values = self.explainer(feature_vector.reshape(1, -1))
        shap_values.feature_names = feature_names
        
        plt.figure(figsize=(10, 6))
        # We use a bar plot of the SHAP values for the single prediction
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        return save_path
