import librosa
import numpy as np

def analyze_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitch = librosa.yin(y, fmin=80, fmax=400)
        pitch_mean = np.nanmean(pitch)
        
        intensity = np.abs(y)
        intensity_mean = np.mean(intensity)
        
        jitter = np.mean(np.abs(np.diff(pitch)))
        shimmer = np.mean(np.abs(np.diff(intensity)))
        
        stress_level = min(100, max(0, 
            (pitch_mean / 200 * 30) + 
            (jitter * 100 * 20) + 
            (shimmer * 50 * 20) +
            (intensity_mean * 30)
        ))
        
        return {
            "stress_level": float(stress_level),
        }
    except Exception:
        return {"stress_level": 50.0}