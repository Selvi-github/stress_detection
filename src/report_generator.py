import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime

class ReportGenerator:
    def __init__(self, reports_dir="reports"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report(self, session_id, metrics_df, shap_plot_path=None):
        if metrics_df.empty:
            print("No metrics to generate report.")
            return None

        # Basic Stats
        avg_hr = metrics_df['hr'].mean()
        avg_hrv = metrics_df['hrv'].mean()
        total_blinks = metrics_df['blinks'].max()
        total_yawns = metrics_df['yawns'].max()
        avg_stress = metrics_df['stress_score'].mean()
        dominant_emotion = metrics_df['dominant_emotion'].mode()[0]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.reports_dir, f"Stress_Report_Session_{session_id}_{timestamp}.pdf")
        
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        
        # Header
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Medical-Style Stress Analysis Report")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(50, height - 100, f"Session ID: {session_id}")
        
        # Physiological Metrics
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 140, "Physiological Indicators (Estimated):")
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 160, f"- Average Heart Rate: {avg_hr:.1f} BPM")
        c.drawString(70, height - 180, f"- Average HRV (SDNN): {avg_hrv:.1f} ms")
        
        # Behavioral Metrics
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 220, "Behavioral & Psychological Indicators:")
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 240, f"- Total Blinks: {total_blinks}")
        c.drawString(70, height - 260, f"- Total Yawns: {total_yawns}")
        c.drawString(70, height - 280, f"- Dominant Emotion: {dominant_emotion}")
        
        # Stress Classification (Medical Clinical Alignment)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 330, f"Average Stress Score: {avg_stress:.1f} / 100")
        
        level = "Normal"
        if avg_stress >= 66:
            level = "Chronic Stress"
        elif avg_stress >= 33:
            level = "Acute Stress"
            
        c.drawString(50, height - 360, f"Clinical Assessment: {level}")
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, height - 380, "*Acute = Short-term situational reaction; Chronic = Persistent abnormal behavior over time")
        
        # SHAP Plot
        if shap_plot_path and os.path.exists(shap_plot_path):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 410, "Explainability (SHAP Analysis of Final Frame):")
            
            try:
                img = ImageReader(shap_plot_path)
                # Plot image
                c.drawImage(img, 50, height - 700, width=500, preserveAspectRatio=True)
            except Exception as e:
                c.setFont("Helvetica", 10)
                c.drawString(50, height - 430, f"Error loading SHAP plot: {e}")
                
        # Footer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 50, "Disclaimer: This report is for research purposes only and not a medical diagnosis.")
        
        c.save()
        return filepath
