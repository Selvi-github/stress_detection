import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class StressDashboard(ctk.CTk):
    def __init__(self, on_closing_callback=None, end_session_callback=None):
        super().__init__()

        self.title("AI-Based Non-Contact Facial Stress Detection")
        self.geometry("1200x800")
        
        self.on_closing_callback = on_closing_callback
        self.end_session_callback = end_session_callback
        
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Configure grid layout
        self.grid_columnconfigure(0, weight=2) # Video feed
        self.grid_columnconfigure(1, weight=1) # Metrics
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Video Frame
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")

        # Metrics Panel
        self.metrics_frame = ctk.CTkFrame(self)
        self.metrics_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(self.metrics_frame, text="Live Metrics", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        self.hr_label = ctk.CTkLabel(self.metrics_frame, text="Heart Rate: -- BPM", font=ctk.CTkFont(size=16))
        self.hr_label.pack(pady=5)
        
        self.hrv_label = ctk.CTkLabel(self.metrics_frame, text="HRV (SDNN): -- ms", font=ctk.CTkFont(size=16))
        self.hrv_label.pack(pady=5)
        
        self.blinks_label = ctk.CTkLabel(self.metrics_frame, text="Blinks: 0", font=ctk.CTkFont(size=16))
        self.blinks_label.pack(pady=5)

        self.yawns_label = ctk.CTkLabel(self.metrics_frame, text="Yawns: 0", font=ctk.CTkFont(size=16))
        self.yawns_label.pack(pady=5)
        
        self.emotion_label = ctk.CTkLabel(self.metrics_frame, text="Emotion: --", font=ctk.CTkFont(size=16))
        self.emotion_label.pack(pady=5)

        # Stress Score Panel
        self.stress_frame = ctk.CTkFrame(self.metrics_frame, fg_color="transparent")
        self.stress_frame.pack(pady=20, fill="x")
        
        self.stress_val_label = ctk.CTkLabel(self.stress_frame, text="Stress Score: --", font=ctk.CTkFont(size=24, weight="bold"))
        self.stress_val_label.pack(pady=5)
        
        self.stress_level_label = ctk.CTkLabel(self.stress_frame, text="Level: --", font=ctk.CTkFont(size=18))
        self.stress_level_label.pack(pady=5)
        
        # End Session Button
        self.end_btn = ctk.CTkButton(self.metrics_frame, text="End Session & Generate Report", command=self._end_session)
        self.end_btn.pack(side="bottom", pady=20)

        # Plot Panel
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.fig, self.ax = plt.subplots(figsize=(4, 2), dpi=100)
        self.ax.set_title("rPPG Signal (Filtered)")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

    def update_video(self, cv_image):
        if cv_image is None:
            return
        # Convert OpenCV BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        # Resize to fit frame
        pil_image = pil_image.resize((800, 600), Image.Resampling.LANCZOS)
        # Convert to CTkImage
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(800, 600))
        
        self.video_label.configure(image=ctk_image)
        self.video_label.image = ctk_image

    def update_metrics(self, hr, hrv, blinks, yawns, emotion, stress_score, stress_level):
        self.hr_label.configure(text=f"Heart Rate: {hr:.1f} BPM")
        self.hrv_label.configure(text=f"HRV (SDNN): {hrv:.1f} ms")
        self.blinks_label.configure(text=f"Blinks: {blinks}")
        self.yawns_label.configure(text=f"Yawns: {yawns}")
        self.emotion_label.configure(text=f"Emotion: {emotion}")
        self.stress_val_label.configure(text=f"Stress Score: {stress_score:.1f}")
        self.stress_level_label.configure(text=f"Level: {stress_level}")
        
        if stress_level == "Chronic Stress":
            self.stress_val_label.configure(text_color="red")
            self.stress_level_label.configure(text_color="red")
        elif stress_level == "Acute Stress":
            self.stress_val_label.configure(text_color="orange")
            self.stress_level_label.configure(text_color="orange")
        else:
            self.stress_val_label.configure(text_color="green")
            self.stress_level_label.configure(text_color="green")

    def update_plot(self, signal_data):
        self.ax.clear()
        self.ax.set_title("rPPG Signal (Filtered)")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        if len(signal_data) > 0:
            self.ax.plot(signal_data, color='blue')
            
        self.canvas.draw()

    def _end_session(self):
        if self.end_session_callback:
            self.end_session_callback()

    def _on_closing(self):
        if self.on_closing_callback:
            self.on_closing_callback()
        self.destroy()
