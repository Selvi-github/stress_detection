import numpy as np
import os
import joblib
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from collections import deque

class StressClassifier:
    def __init__(self, model_path="models/stress_rf.pkl", persistence_window=10):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        # Track historical score window to distinguish Acute (transient) vs Chronic (persistent) stress
        self.history = deque(maxlen=persistence_window)
        
        # We use a Regressor to output a continuous Stress Score (0-100)
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self._train_mock_model()

    def _train_mock_model(self):
        """
        Train a dummy model so the system runs out-of-the-box.
        In a real scenario, this is trained on WESAD or UBFC datasets.
        """
        X_dummy = np.random.rand(100, 16)
        
        # Mock logic: High HR, Low HRV, High Yawns, High Angry/Fear -> High Stress
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
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def clear_history(self):
        self.history.clear()

    def predict(self, feature_vector, feature_names):
        """
        Predict stress score and return SHAP explanation.
        Categorizes according to medical clinical standards:
        - Normal / Baseline: score < 33
        - Acute Stress: transient situational stress spike (33 <= score < 66)
        - Chronic Stress: persistent, long-duration abnormal behavior (score >= 66)
        """
        if feature_vector is None or len(feature_vector) == 0:
            return 0.0, "Normal", None

        # Predict score (0 - 100)
        score = self.model.predict(feature_vector.reshape(1, -1))[0]
        score = np.clip(score, 0, 100)
        
        # Track history for temporal persistence analysis
        self.history.append(score)

        # Clinical Medical Categorization:
        # Score < 33: Normal / Baseline
        # Score 33 - 65.9: Acute Stress
        # Score >= 66: Chronic Stress
        if score < 33.0:
            level = "Normal"
        elif score < 66.0:
            level = "Acute Stress"
        else:
            level = "Chronic Stress"

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(feature_vector.reshape(1, -1))
        
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
