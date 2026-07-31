import numpy as np
from scipy import signal
from collections import deque
import time

class RPPGEngine:
    def __init__(self, fps=30, window_size_sec=10):
        self.fps = fps
        self.window_size = fps * window_size_sec
        self.buffer_r = deque(maxlen=self.window_size)
        self.buffer_g = deque(maxlen=self.window_size)
        self.buffer_b = deque(maxlen=self.window_size)
        self.times = deque(maxlen=self.window_size)
        self.start_time = None
        
        self.last_hr = 0
        self.last_hrv = 0

    def add_frame_mean(self, rgb_mean):
        """
        Add the mean RGB value of an ROI for the current frame.
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        r, g, b = rgb_mean
        self.buffer_r.append(r)
        self.buffer_g.append(g)
        self.buffer_b.append(b)
        self.times.append(time.time() - self.start_time)

    def is_ready(self):
        """
        Check if we have enough data to compute HR.
        We need at least 3 seconds of data (3 * fps frames).
        """
        return len(self.buffer_g) >= self.fps * 3

    def get_signal_green(self):
        """
        Basic GREEN channel method.
        """
        return np.array(self.buffer_g)

    def get_signal_pos(self):
        """
        Plane-Orthogonal-to-Skin (POS) algorithm.
        Reference: Wang et al. 2016, "Algorithmic Principles of Remote PPG"
        """
        r = np.array(self.buffer_r)
        g = np.array(self.buffer_g)
        b = np.array(self.buffer_b)

        # Normalize by mean
        r_n = r / (np.mean(r) + 1e-6)
        g_n = g / (np.mean(g) + 1e-6)
        b_n = b / (np.mean(b) + 1e-6)

        # Projection
        x = r_n - g_n
        y = r_n + g_n - 2 * b_n

        # Alpha tuning
        alpha = np.std(x) / (np.std(y) + 1e-6)
        
        h = x + alpha * y
        return h

    def filter_signal(self, sig, min_hz=0.75, max_hz=2.5):
        """
        Bandpass filter to isolate human heart rate (45 - 150 BPM).
        """
        # Calculate actual FPS based on timestamps if available
        if len(self.times) > 1:
            dt = np.diff(self.times)
            actual_fps = 1.0 / np.mean(dt)
        else:
            actual_fps = self.fps

        nyq = 0.5 * actual_fps
        low = min_hz / nyq
        high = max_hz / nyq
        
        # Ensure bounds
        low = max(0.01, min(0.99, low))
        high = max(0.02, min(0.99, high))

        b, a = signal.butter(3, [low, high], btype='band')
        try:
            filtered = signal.filtfilt(b, a, sig)
        except ValueError:
            # If sequence is too short for filtfilt
            filtered = sig
        return filtered

    def estimate_heart_rate(self, method='POS'):
        if not self.is_ready():
            return 0, 0, []

        if method == 'POS':
            raw_sig = self.get_signal_pos()
        else:
            raw_sig = self.get_signal_green()

        filtered_sig = self.filter_signal(raw_sig)

        # Compute FFT to find dominant frequency
        actual_fps = self.fps
        if len(self.times) > 1:
            actual_fps = 1.0 / np.mean(np.diff(self.times))

        freqs = np.fft.rfftfreq(len(filtered_sig), 1.0 / actual_fps)
        fft_mag = np.abs(np.fft.rfft(filtered_sig))

        # Restrict to human HR range (0.75 Hz to 2.5 Hz)
        valid_idx = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
        if len(valid_idx) == 0:
            return self.last_hr, self.last_hrv, filtered_sig

        best_freq = freqs[valid_idx[np.argmax(fft_mag[valid_idx])]]
        hr_bpm = best_freq * 60.0

        # Simple HRV proxy: standard deviation of peak-to-peak intervals
        peaks, _ = signal.find_peaks(filtered_sig, distance=actual_fps/2.5) # min distance corresponding to max HR
        if len(peaks) > 1:
            intervals = np.diff(peaks) / actual_fps
            hrv_sdnn = np.std(intervals) * 1000 # in ms
        else:
            hrv_sdnn = 0

        # Smooth output
        if self.last_hr != 0:
            hr_bpm = 0.8 * self.last_hr + 0.2 * hr_bpm

        self.last_hr = hr_bpm
        self.last_hrv = hrv_sdnn

        return hr_bpm, hrv_sdnn, filtered_sig

