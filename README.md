# AI-Based Non-Contact Facial Stress Detection System

This is a real-time, research-grade stress detection system that estimates physiological, behavioral, and psychological indicators using only a standard RGB webcam on the CPU. No wearable sensors or GPUs are required.

## Features
- **Face Detection & Tracking**: Tracks employee faces across frames.
- **Landmark & ROI Extraction**: Extracts 468 MediaPipe landmarks to identify forehead and cheeks.
- **Remote PPG (rPPG)**: Estimates Heart Rate (BPM) and HRV from facial micro-color changes using the POS and GREEN methods.
- **Behavioral Analysis**: Calculates Blink Rate (EAR), Yawning (MAR), and Head Pose (Pitch, Yaw, Roll).
- **Emotion Analysis**: Analyzes facial muscle tension to estimate emotional state.
- **Stress Classification**: Fuses multi-modal features and uses a Random Forest Regressor to output a Stress Score (0-100).
- **Explainable AI (SHAP)**: Explains the stress prediction by showing which features contributed the most.
- **Medical Report Generation**: Outputs a PDF report summarizing the session.

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

This will open a dashboard showing the live webcam feed and real-time metrics. 

When you are done, click **End Session & Generate Report**. The system will save a PDF report with statistics and a SHAP explainability plot in the `reports/` folder. All raw data is logged in a local SQLite database (`data/stress_data.db`).

## Notes on CPU Performance
- The application separates UI updates and heavy MediaPipe/rPPG computation into different threads to maintain responsiveness.
- Ensure good lighting for accurate rPPG estimation.
