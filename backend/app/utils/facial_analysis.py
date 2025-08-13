import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

FAU_MAPPING = {
    "AU1": [18, 19, 20, 21],
    "AU2": [22, 23, 24, 25],
    "AU4": [17, 18, 26, 27],
    "AU5": [36, 37, 38, 39, 40, 41],
    "AU6": [41, 42, 43, 44, 45, 46, 47],
    "AU9": [31, 32, 33, 34, 35],
    "AU12": [48, 49, 50, 51, 52, 53, 54, 59, 60],
    "AU15": [55, 56, 57, 58],
    "AU20": [61, 62, 63, 67],
    "AU25": [60, 61, 62, 63, 64, 65, 66, 67]
}

def extract_landmarks(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        landmarks = []
        
        for face in faces:
            shape = predictor(gray, face)
            for i in range(68):
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append([x, y])
        
        return np.array(landmarks) if landmarks else None
    except Exception:
        return None

def detect_emotion(landmarks):
    if landmarks is None or len(landmarks) < 68:
        return "neutral"
    
    try:
        left_eyebrow = landmarks[21]
        right_eyebrow = landmarks[22]
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        upper_lip = landmarks[51]
        lower_lip = landmarks[57]
        
        eye_distance = abs(left_eye[0] - right_eye[0])
        eyebrow_distance = abs(left_eyebrow[0] - right_eyebrow[0])
        mouth_openness = abs(upper_lip[1] - lower_lip[1])
        
        if mouth_openness > 20 and eyebrow_distance > eye_distance:
            return "surprise"
        elif eyebrow_distance < eye_distance - 10 and mouth_openness < 10:
            return "angry"
        else:
            return "neutral"
    except Exception:
        return "neutral"

def analyze_frame(frame):
    landmarks = extract_landmarks(frame)
    emotion = detect_emotion(landmarks)
    
    return {
        "dominant_emotion": emotion,
        "micro_expressions": [],
    }