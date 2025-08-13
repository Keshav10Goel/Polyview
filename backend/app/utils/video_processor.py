import cv2
import os
from datetime import datetime
from .facial_analysis import analyze_frame

def extract_frames(video_path, sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return frames

def process_video(video_path):
    frames = extract_frames(video_path)
    results = {
        'emotions': [],
        'micro_expressions': [],
    }
    
    for frame in frames:
        frame_result = analyze_frame(frame)
        results['emotions'].append(frame_result['dominant_emotion'])
        results['micro_expressions'].extend(frame_result['micro_expressions'])
    
    return results

def save_uploaded_file(file, upload_folder):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"analysis_{timestamp}_{file.filename}"
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    return file_path