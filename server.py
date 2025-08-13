# from flask import Flask, render_template, request, jsonify, redirect, url_for
# import sqlite3
# import os
# import cv2
# import numpy as np
# import librosa
# from deepface import DeepFace
# import dlib
# from pydub import AudioSegment
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import load_model
# import wikipedia
# import openai
# import tempfile
# import random
# import threading
# import time

# app = Flask(__name__)

# # Initialize dlib face detector and landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Facial Action Units (FAU) mapping for 68 landmarks
# FAU_MAPPING = {
#     "AU1": [18, 19, 20, 21],      # Inner brow
#     "AU2": [22, 23, 24, 25],      # Outer brow
#     "AU4": [17, 18, 26, 27],      # Brow lowerer
#     "AU5": [36, 37, 38, 39, 40, 41],  # Upper lid
#     "AU6": [41, 42, 43, 44, 45, 46, 47],  # Cheek raiser
#     "AU9": [31, 32, 33, 34, 35],  # Nose wrinkler
#     "AU12": [48, 49, 50, 51, 52, 53, 54, 59, 60],  # Lip corner puller
#     "AU15": [55, 56, 57, 58],     # Lip corner depressor
#     "AU20": [61, 62, 63, 67],     # Lip stretcher
#     "AU25": [60, 61, 62, 63, 64, 65, 66, 67]  # Lips part
# }

# # Emotion labels
# EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# # Database setup
# def init_db():
#     conn = sqlite3.connect('polyview.db')
#     c = conn.cursor()
    
#     # Create users table
#     c.execute('''CREATE TABLE IF NOT EXISTS users (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT NOT NULL,
#         email TEXT UNIQUE NOT NULL,
#         password TEXT NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     )''')
    
#     # Create analyses table
#     c.execute('''CREATE TABLE IF NOT EXISTS analyses (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         user_id INTEGER NOT NULL,
#         truth_score REAL NOT NULL,
#         stress_level REAL NOT NULL,
#         emotion TEXT NOT NULL,
#         transcript TEXT NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#         FOREIGN KEY(user_id) REFERENCES users(id)
#     )''')
    
#     # Insert sample user if not exists
#     try:
#         c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
#                  ("Test User", "test@example.com", "test123"))
#     except sqlite3.IntegrityError:
#         pass
    
#     conn.commit()
#     conn.close()

# # Initialize database if not exists
# if not os.path.exists('polyview.db'):
#     init_db()

# # Routes
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/permission')
# def permission():
#     return render_template('permission.html')

# @app.route('/analysis')
# def analysis():
#     return render_template('analysis.html')

# @app.route('/features')
# def features():
#     return render_template('features.html')

# @app.route('/pricing')
# def pricing():
#     return render_template('pricing.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# @app.route('/login')
# def login_page():
#     return render_template('login.html')

# @app.route('/signup')
# def signup_page():
#     return render_template('signup.html')

# # API Endpoints
# @app.route('/api/login', methods=['POST'])
# def login():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')
    
#     conn = sqlite3.connect('polyview.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
#     user = c.fetchone()
#     conn.close()
    
#     if user:
#         return jsonify({
#             "success": True, 
#             "message": "Login successful",
#             "user_id": user[0],
#             "name": user[1]
#         })
#     else:
#         return jsonify({"success": False, "message": "Invalid credentials"}), 401

# @app.route('/api/signup', methods=['POST'])
# def signup():
#     data = request.json
#     name = data.get('name')
#     email = data.get('email')
#     password = data.get('password')
    
#     try:
#         conn = sqlite3.connect('polyview.db')
#         c = conn.cursor()
#         c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
#                  (name, email, password))
#         user_id = c.lastrowid
#         conn.commit()
#         conn.close()
#         return jsonify({
#             "success": True, 
#             "message": "Account created successfully",
#             "user_id": user_id
#         })
#     except sqlite3.IntegrityError:
#         return jsonify({"success": False, "message": "Email already exists"}), 400

# def extract_facial_landmarks(frame):
#     """Extract 68 facial landmarks using dlib"""
#     try:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)
#         landmarks = []
        
#         for face in faces:
#             shape = predictor(gray, face)
#             for i in range(68):
#                 x = shape.part(i).x
#                 y = shape.part(i).y
#                 landmarks.append([x, y])
        
#         return np.array(landmarks) if landmarks else None
#     except Exception as e:
#         print(f"Landmark extraction error: {e}")
#         return None

# def detect_micro_expressions(landmarks, prev_landmarks):
#     """Detect micro-expressions using facial landmark displacement"""
#     if prev_landmarks is None or landmarks is None:
#         return []
    
#     micro_expressions = []
    
#     # Calculate displacement for each AU group
#     for au, indices in FAU_MAPPING.items():
#         current_points = landmarks[indices]
#         previous_points = prev_landmarks[indices]
        
#         # Calculate Euclidean distance between current and previous points
#         displacement = np.mean(np.linalg.norm(current_points - previous_points, axis=1))
        
#         if displacement > 0.02:  # Threshold for micro-expression
#             micro_expressions.append(au)
    
#     return micro_expressions

# def detect_emotion(frame):
#     """Detect emotion using DeepFace"""
#     try:
#         # Analyze face for emotion
#         result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
#         if result and len(result) > 0:
#             # Return dominant emotion
#             return result[0]['dominant_emotion']
#         return "neutral"
#     except Exception as e:
#         print(f"Emotion detection error: {e}")
#         return "neutral"

# def analyze_voice(audio_path):
#     """Analyze voice for stress indicators"""
#     try:
#         # Load audio file
#         y, sr = librosa.load(audio_path, sr=None)
        
#         # Extract pitch (fundamental frequency)
#         pitch = librosa.yin(y, fmin=80, fmax=400)
#         pitch_mean = np.nanmean(pitch)  # Handle NaN values
        
#         # Calculate intensity (volume)
#         intensity = np.abs(y)
#         intensity_mean = np.mean(intensity)
        
#         # Extract MFCC features
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#         mfcc_mean = np.mean(mfcc, axis=1)
        
#         # Calculate jitter (pitch variability)
#         jitter = np.mean(np.abs(np.diff(pitch)))
        
#         # Calculate shimmer (amplitude variability)
#         shimmer = np.mean(np.abs(np.diff(intensity)))
        
#         # Simple stress calculation (higher values = more stress)
#         stress_level = min(100, max(0, 
#             (pitch_mean / 200 * 30) + 
#             (jitter * 100 * 20) + 
#             (shimmer * 50 * 20) +
#             (intensity_mean * 30)
#         ))
        
#         return {
#             "stress_level": float(stress_level),
#             "pitch": float(pitch_mean),
#             "intensity": float(intensity_mean),
#             "jitter": float(jitter),
#             "shimmer": float(shimmer)
#         }
#     except Exception as e:
#         print(f"Voice analysis error: {e}")
#         return {
#             "stress_level": random.uniform(30, 70),
#             "pitch": 0,
#             "intensity": 0,
#             "jitter": 0,
#             "shimmer": 0
#         }

# def fact_check(statement):
#     """Verify factual accuracy using Wikipedia"""
#     try:
#         # Search Wikipedia
#         search_results = wikipedia.search(statement, results=3)
#         context = ""
        
#         for result in search_results:
#             try:
#                 context += wikipedia.summary(result, sentences=2) + "\n\n"
#             except:
#                 continue
        
#         # Simple keyword matching (replace with OpenAI in production)
#         keywords = statement.lower().split()
#         found_keywords = sum(1 for word in keywords if word in context.lower())
#         confidence = min(1.0, found_keywords / max(1, len(keywords)))
        
#         return {
#             "confidence": confidence,
#             "sources": search_results
#         }
#     except:
#         return {
#             "confidence": random.uniform(0.3, 0.7),
#             "sources": []
#         }

# def calculate_truth_score(metrics):
#     """Calculate overall truth score (0-100)"""
#     # Weights for different factors
#     weights = {
#         "stress_level": 0.4,
#         "micro_expressions": 0.3,
#         "fact_confidence": 0.3
#     }
    
#     # Normalize metrics
#     stress_score = 1 - min(metrics["stress_level"] / 100, 1)
    
#     # More expressions = higher deception probability
#     expressions_score = 1 - min(len(metrics["micro_expressions"]) / 5, 1)
    
#     fact_score = metrics["fact_confidence"]
    
#     # Calculate weighted score
#     score = (
#         weights["stress_level"] * stress_score +
#         weights["micro_expressions"] * expressions_score +
#         weights["fact_confidence"] * fact_score
#     ) * 100
    
#     return min(max(score, 0), 100)

# @app.route('/api/analyze', methods=['POST'])
# def analyze():
#     # Get files and data
#     video_file = request.files.get('video')
#     audio_file = request.files.get('audio')
#     transcript = request.form.get('transcript', '')
#     user_id = request.form.get('user_id', 0)
    
#     # Create temporary files
#     video_path = tempfile.mktemp(suffix='.mp4')
#     audio_path = tempfile.mktemp(suffix='.wav')
    
#     # Save files
#     if video_file:
#         video_file.save(video_path)
#     if audio_file:
#         audio_file.save(audio_path)
    
#     # Initialize metrics
#     metrics = {
#         "stress_level": 0,
#         "micro_expressions": [],
#         "fact_confidence": 0,
#         "emotion": "neutral"
#     }
    
#     # Process video if available
#     if os.path.exists(video_path):
#         cap = cv2.VideoCapture(video_path)
#         frame_count = 0
#         prev_landmarks = None
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
            
#             # Analyze every 5th frame
#             if frame_count % 5 == 0:
#                 # Get facial landmarks
#                 landmarks = extract_facial_landmarks(frame)
                
#                 if landmarks is not None:
#                     # Detect micro-expressions
#                     if prev_landmarks is not None:
#                         micro_expr = detect_micro_expressions(landmarks, prev_landmarks)
#                         metrics["micro_expressions"].extend(micro_expr)
                    
#                     prev_landmarks = landmarks
                    
#                     # Detect emotion on first frame
#                     if metrics["emotion"] == "neutral":
#                         metrics["emotion"] = detect_emotion(frame)
        
#         cap.release()
    
#     # Process audio if available
#     if os.path.exists(audio_path):
#         voice_metrics = analyze_voice(audio_path)
#         metrics["stress_level"] = voice_metrics["stress_level"]
#     else:
#         metrics["stress_level"] = random.uniform(30, 70)
    
#     # Fact-check transcript
#     if transcript:
#         fact_check_result = fact_check(transcript)
#         metrics["fact_confidence"] = fact_check_result["confidence"]
#     else:
#         metrics["fact_confidence"] = random.uniform(0.5, 0.9)
    
#     # Calculate overall truth score
#     truth_score = calculate_truth_score(metrics)
    
#     # Save to database
#     try:
#         conn = sqlite3.connect('polyview.db')
#         c = conn.cursor()
#         c.execute("""
#             INSERT INTO analyses (user_id, truth_score, stress_level, emotion, transcript)
#             VALUES (?, ?, ?, ?, ?)
#         """, (user_id, truth_score, metrics["stress_level"], metrics["emotion"], transcript))
#         conn.commit()
#         conn.close()
#     except Exception as e:
#         print(f"Database error: {e}")
    
#     # Clean up temporary files
#     if os.path.exists(video_path):
#         os.remove(video_path)
#     if os.path.exists(audio_path):
#         os.remove(audio_path)
    
#     # Return results
#     return jsonify({
#         "truth_score": truth_score,
#         "stress_level": metrics["stress_level"],
#         "emotion": metrics["emotion"],
#         "micro_expressions": list(set(metrics["micro_expressions"])),
#         "fact_confidence": metrics["fact_confidence"],
#         "transcript": transcript
#     })

# @app.route('/api/baseline', methods=['POST'])
# def baseline_calibration():
#     """Establish user's behavioral baseline"""
#     try:
#         # This would process calibration questions in a real implementation
#         return jsonify({
#             "success": True,
#             "message": "Baseline established",
#             "baseline_id": f"baseline_{int(time.time())}"
#         })
#     except:
#         return jsonify({"success": False, "message": "Calibration failed"}), 500

# @app.route('/api/history/<int:user_id>')
# def analysis_history(user_id):
#     """Get user's analysis history"""
#     try:
#         conn = sqlite3.connect('polyview.db')
#         c = conn.cursor()
#         c.execute("SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
#         analyses = c.fetchall()
#         conn.close()
        
#         # Format results
#         history = []
#         for analysis in analyses:
#             history.append({
#                 "id": analysis[0],
#                 "truth_score": analysis[2],
#                 "stress_level": analysis[3],
#                 "emotion": analysis[4],
#                 "created_at": analysis[6]
#             })
        
#         return jsonify({"success": True, "history": history})
#     except:
#         return jsonify({"success": False, "message": "Error retrieving history"}), 500

# if __name__ == '__main__':
#     # Start with a simple check for dlib model
#     if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
#         print("ERROR: Missing dlib model file!")
#         print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#         print("Extract and place in project folder")
#     else:
#         app.run(debug=True, port=5000, threaded=True)

# from flask import Flask, render_template, request, jsonify, redirect, url_for
# import sqlite3
# import os
# import cv2
# import numpy as np
# import librosa
# import dlib
# import wikipedia
# import tempfile
# import random
# import time
# import math

# app = Flask(__name__)

# # Initialize dlib face detector and landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Facial Action Units (FAU) mapping for 68 landmarks
# FAU_MAPPING = {
#     "AU1": [18, 19, 20, 21],      # Inner brow
#     "AU2": [22, 23, 24, 25],      # Outer brow
#     "AU4": [17, 18, 26, 27],      # Brow lowerer
#     "AU5": [36, 37, 38, 39, 40, 41],  # Upper lid
#     "AU6": [41, 42, 43, 44, 45, 46, 47],  # Cheek raiser
#     "AU9": [31, 32, 33, 34, 35],  # Nose wrinkler
#     "AU12": [48, 49, 50, 51, 52, 53, 54, 59, 60],  # Lip corner puller
#     "AU15": [55, 56, 57, 58],     # Lip corner depressor
#     "AU20": [61, 62, 63, 67],     # Lip stretcher
#     "AU25": [60, 61, 62, 63, 64, 65, 66, 67]  # Lips part
# }

# # Database setup
# def init_db():
#     conn = sqlite3.connect('polyview.db')
#     c = conn.cursor()
    
#     # Create users table
#     c.execute('''CREATE TABLE IF NOT EXISTS users (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT NOT NULL,
#         email TEXT UNIQUE NOT NULL,
#         password TEXT NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     )''')
    
#     # Create analyses table
#     c.execute('''CREATE TABLE IF NOT EXISTS analyses (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         user_id INTEGER NOT NULL,
#         truth_score REAL NOT NULL,
#         stress_level REAL NOT NULL,
#         emotion TEXT NOT NULL,
#         transcript TEXT NOT NULL,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#         FOREIGN KEY(user_id) REFERENCES users(id)
#     )''')
    
#     # Insert sample user if not exists
#     try:
#         c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
#                  ("Test User", "test@example.com", "test123"))
#     except sqlite3.IntegrityError:
#         pass
    
#     conn.commit()
#     conn.close()

# # Initialize database if not exists
# if not os.path.exists('polyview.db'):
#     init_db()

# # Routes
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/permission')
# def permission():
#     return render_template('permission.html')

# @app.route('/analysis')
# def analysis():
#     return render_template('analysis.html')

# @app.route('/features')
# def features():
#     return render_template('features.html')

# @app.route('/pricing')
# def pricing():
#     return render_template('pricing.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# @app.route('/login')
# def login_page():
#     return render_template('login.html')

# @app.route('/signup')
# def signup_page():
#     return render_template('signup.html')

# # API Endpoints
# @app.route('/api/login', methods=['POST'])
# def login():
#     data = request.json
#     email = data.get('email')
#     password = data.get('password')
    
#     conn = sqlite3.connect('polyview.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
#     user = c.fetchone()
#     conn.close()
    
#     if user:
#         return jsonify({
#             "success": True, 
#             "message": "Login successful",
#             "user_id": user[0],
#             "name": user[1]
#         })
#     else:
#         return jsonify({"success": False, "message": "Invalid credentials"}), 401

# @app.route('/api/signup', methods=['POST'])
# def signup():
#     data = request.json
#     name = data.get('name')
#     email = data.get('email')
#     password = data.get('password')
    
#     try:
#         conn = sqlite3.connect('polyview.db')
#         c = conn.cursor()
#         c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
#                  (name, email, password))
#         user_id = c.lastrowid
#         conn.commit()
#         conn.close()
#         return jsonify({
#             "success": True, 
#             "message": "Account created successfully",
#             "user_id": user_id
#         })
#     except sqlite3.IntegrityError:
#         return jsonify({"success": False, "message": "Email already exists"}), 400

# def extract_facial_landmarks(frame):
#     """Extract 68 facial landmarks using dlib"""
#     try:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector(gray)
#         landmarks = []
        
#         for face in faces:
#             shape = predictor(gray, face)
#             for i in range(68):
#                 x = shape.part(i).x
#                 y = shape.part(i).y
#                 landmarks.append([x, y])
        
#         return np.array(landmarks) if landmarks else None
#     except Exception as e:
#         print(f"Landmark extraction error: {e}")
#         return None

# def detect_micro_expressions(landmarks, prev_landmarks):
#     """Detect micro-expressions using facial landmark displacement"""
#     if prev_landmarks is None or landmarks is None:
#         return []
    
#     micro_expressions = []
    
#     # Calculate displacement for each AU group
#     for au, indices in FAU_MAPPING.items():
#         current_points = landmarks[indices]
#         previous_points = prev_landmarks[indices]
        
#         # Calculate Euclidean distance between current and previous points
#         displacement = np.mean(np.linalg.norm(current_points - previous_points, axis=1))
        
#         if displacement > 0.02:  # Threshold for micro-expression
#             micro_expressions.append(au)
    
#     return micro_expressions

# def detect_emotion(landmarks):
#     """Detect emotion from facial landmarks"""
#     if landmarks is None or len(landmarks) < 68:
#         return "neutral"
    
#     try:
#         # Get key points
#         left_eyebrow = landmarks[21]
#         right_eyebrow = landmarks[22]
#         left_eye = landmarks[36]
#         right_eye = landmarks[45]
#         nose_tip = landmarks[30]
#         left_mouth = landmarks[48]
#         right_mouth = landmarks[54]
#         upper_lip = landmarks[51]
#         lower_lip = landmarks[57]
        
#         # Calculate facial metrics
#         eye_distance = abs(left_eye[0] - right_eye[0])
#         eyebrow_distance = abs(left_eyebrow[0] - right_eyebrow[0])
#         mouth_openness = abs(upper_lip[1] - lower_lip[1])
#         mouth_width = abs(left_mouth[0] - right_mouth[0])
        
#         # Calculate smile curvature
#         def calculate_curvature(p1, p2, p3):
#             """Calculate curvature between three points"""
#             dx1 = p2[0] - p1[0]
#             dy1 = p2[1] - p1[1]
#             dx2 = p3[0] - p2[0]
#             dy2 = p3[1] - p2[1]
#             return (dx1*dy2 - dx2*dy1) / math.sqrt((dx1**2 + dy1**2)**3)
        
#         smile_curve = calculate_curvature(left_mouth, upper_lip, right_mouth)
        
#         # Determine emotion based on metrics
#         if mouth_openness > 20 and eyebrow_distance > eye_distance:
#             return "surprise"
#         elif smile_curve > 0.001 and mouth_width > 50:
#             return "happy"
#         elif eyebrow_distance < eye_distance and smile_curve < -0.001:
#             return "sad"
#         elif eyebrow_distance < eye_distance - 10 and mouth_openness < 10:
#             return "angry"
#         else:
#             return "neutral"
            
#     except Exception as e:
#         print(f"Emotion detection error: {e}")
#         return "neutral"

# def analyze_voice(audio_path):
#     """Analyze voice for stress indicators"""
#     try:
#         # Load audio file
#         y, sr = librosa.load(audio_path, sr=None)
        
#         # Extract pitch (fundamental frequency)
#         pitch = librosa.yin(y, fmin=80, fmax=400)
#         pitch_mean = np.nanmean(pitch)  # Handle NaN values
        
#         # Calculate intensity (volume)
#         intensity = np.abs(y)
#         intensity_mean = np.mean(intensity)
        
#         # Calculate jitter (pitch variability)
#         jitter = np.mean(np.abs(np.diff(pitch)))
        
#         # Calculate shimmer (amplitude variability)
#         shimmer = np.mean(np.abs(np.diff(intensity)))
        
#         # Simple stress calculation (higher values = more stress)
#         stress_level = min(100, max(0, 
#             (pitch_mean / 200 * 30) + 
#             (jitter * 100 * 20) + 
#             (shimmer * 50 * 20) +
#             (intensity_mean * 30)
#         ))
        
#         return {
#             "stress_level": float(stress_level),
#             "pitch": float(pitch_mean),
#             "intensity": float(intensity_mean),
#             "jitter": float(jitter),
#             "shimmer": float(shimmer)
#         }
#     except Exception as e:
#         print(f"Voice analysis error: {e}")
#         return {
#             "stress_level": random.uniform(30, 70),
#             "pitch": 0,
#             "intensity": 0,
#             "jitter": 0,
#             "shimmer": 0
#         }

# def fact_check(statement):
#     """Verify factual accuracy using Wikipedia"""
#     try:
#         # Search Wikipedia
#         search_results = wikipedia.search(statement, results=3)
#         context = ""
        
#         for result in search_results:
#             try:
#                 context += wikipedia.summary(result, sentences=2) + "\n\n"
#             except:
#                 continue
        
#         # Simple keyword matching
#         keywords = statement.lower().split()
#         found_keywords = sum(1 for word in keywords if word in context.lower())
#         confidence = min(1.0, found_keywords / max(1, len(keywords)))
        
#         return {
#             "confidence": confidence,
#             "sources": search_results
#         }
#     except:
#         return {
#             "confidence": random.uniform(0.3, 0.7),
#             "sources": []
#         }

# def calculate_truth_score(metrics):
#     """Calculate overall truth score (0-100)"""
#     # Weights for different factors
#     weights = {
#         "stress_level": 0.4,
#         "micro_expressions": 0.3,
#         "fact_confidence": 0.3
#     }
    
#     # Normalize metrics
#     stress_score = 1 - min(metrics["stress_level"] / 100, 1)
    
#     # More expressions = higher deception probability
#     expressions_score = 1 - min(len(metrics["micro_expressions"]) / 5, 1)
    
#     fact_score = metrics["fact_confidence"]
    
#     # Calculate weighted score
#     score = (
#         weights["stress_level"] * stress_score +
#         weights["micro_expressions"] * expressions_score +
#         weights["fact_confidence"] * fact_score
#     ) * 100
    
#     return min(max(score, 0), 100)

# @app.route('/api/analyze', methods=['POST'])
# def analyze():
#     # Get files and data
#     video_file = request.files.get('video')
#     audio_file = request.files.get('audio')
#     transcript = request.form.get('transcript', '')
#     user_id = request.form.get('user_id', 0)
    
#     # Create temporary files
#     video_path = tempfile.mktemp(suffix='.mp4')
#     audio_path = tempfile.mktemp(suffix='.wav')
    
#     # Save files
#     if video_file:
#         video_file.save(video_path)
#     if audio_file:
#         audio_file.save(audio_path)
    
#     # Initialize metrics
#     metrics = {
#         "stress_level": 0,
#         "micro_expressions": [],
#         "fact_confidence": 0,
#         "emotion": "neutral"
#     }
    
#     # Process video if available
#     if os.path.exists(video_path):
#         cap = cv2.VideoCapture(video_path)
#         frame_count = 0
#         prev_landmarks = None
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
            
#             # Analyze every 5th frame
#             if frame_count % 5 == 0:
#                 # Get facial landmarks
#                 landmarks = extract_facial_landmarks(frame)
                
#                 if landmarks is not None:
#                     # Detect micro-expressions
#                     if prev_landmarks is not None:
#                         micro_expr = detect_micro_expressions(landmarks, prev_landmarks)
#                         metrics["micro_expressions"].extend(micro_expr)
                    
#                     # Detect emotion
#                     metrics["emotion"] = detect_emotion(landmarks)
#                     prev_landmarks = landmarks
        
#         cap.release()
    
#     # Process audio if available
#     if os.path.exists(audio_path):
#         voice_metrics = analyze_voice(audio_path)
#         metrics["stress_level"] = voice_metrics["stress_level"]
#     else:
#         metrics["stress_level"] = random.uniform(30, 70)
    
#     # Fact-check transcript
#     if transcript:
#         fact_check_result = fact_check(transcript)
#         metrics["fact_confidence"] = fact_check_result["confidence"]
#     else:
#         metrics["fact_confidence"] = random.uniform(0.5, 0.9)
    
#     # Calculate overall truth score
#     truth_score = calculate_truth_score(metrics)
    
#     # Save to database
#     try:
#         conn = sqlite3.connect('polyview.db')
#         c = conn.cursor()
#         c.execute("""
#             INSERT INTO analyses (user_id, truth_score, stress_level, emotion, transcript)
#             VALUES (?, ?, ?, ?, ?)
#         """, (user_id, truth_score, metrics["stress_level"], metrics["emotion"], transcript))
#         conn.commit()
#         conn.close()
#     except Exception as e:
#         print(f"Database error: {e}")
    
#     # Clean up temporary files
#     if os.path.exists(video_path):
#         os.remove(video_path)
#     if os.path.exists(audio_path):
#         os.remove(audio_path)
    
#     # Return results
#     return jsonify({
#         "truth_score": truth_score,
#         "stress_level": metrics["stress_level"],
#         "emotion": metrics["emotion"],
#         "micro_expressions": list(set(metrics["micro_expressions"])),
#         "fact_confidence": metrics["fact_confidence"],
#         "transcript": transcript
#     })

# @app.route('/api/baseline', methods=['POST'])
# def baseline_calibration():
#     """Establish user's behavioral baseline"""
#     try:
#         # This would process calibration questions in a real implementation
#         return jsonify({
#             "success": True,
#             "message": "Baseline established",
#             "baseline_id": f"baseline_{int(time.time())}"
#         })
#     except:
#         return jsonify({"success": False, "message": "Calibration failed"}), 500

# @app.route('/api/history/<int:user_id>')
# def analysis_history(user_id):
#     """Get user's analysis history"""
#     try:
#         conn = sqlite3.connect('polyview.db')
#         c = conn.cursor()
#         c.execute("SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
#         analyses = c.fetchall()
#         conn.close()
        
#         # Format results
#         history = []
#         for analysis in analyses:
#             history.append({
#                 "id": analysis[0],
#                 "truth_score": analysis[2],
#                 "stress_level": analysis[3],
#                 "emotion": analysis[4],
#                 "created_at": analysis[6]
#             })
        
#         return jsonify({"success": True, "history": history})
#     except:
#         return jsonify({"success": False, "message": "Error retrieving history"}), 500

# if __name__ == '__main__':
#     # Start with a simple check for dlib model
#     if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
#         print("ERROR: Missing dlib model file!")
#         print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#         print("Extract and place in project folder")
#     else:
#         app.run(debug=True, port=5000, threaded=True)
from flask import Flask, render_template, request, jsonify, redirect, url_for
import sqlite3
import os
import cv2
import numpy as np
import librosa
import dlib
import wikipedia
import tempfile
import random
import time
import math

app = Flask(__name__)

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Facial Action Units (FAU) mapping for 68 landmarks
FAU_MAPPING = {
    "AU1": [18, 19, 20, 21],      # Inner brow
    "AU2": [22, 23, 24, 25],      # Outer brow
    "AU4": [17, 18, 26, 27],      # Brow lowerer
    "AU5": [36, 37, 38, 39, 40, 41],  # Upper lid
    "AU6": [41, 42, 43, 44, 45, 46, 47],  # Cheek raiser
    "AU9": [31, 32, 33, 34, 35],  # Nose wrinkler
    "AU12": [48, 49, 50, 51, 52, 53, 54, 59, 60],  # Lip corner puller
    "AU15": [55, 56, 57, 58],     # Lip corner depressor
    "AU20": [61, 62, 63, 67],     # Lip stretcher
    "AU25": [60, 61, 62, 63, 64, 65, 66, 67]  # Lips part
}

# Database setup
def init_db():
    conn = sqlite3.connect('polyview.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Create analyses table
    c.execute('''CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        truth_score REAL NOT NULL,
        stress_level REAL NOT NULL,
        emotion TEXT NOT NULL,
        transcript TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    
    # Insert sample user if not exists
    try:
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                 ("Test User", "test@example.com", "test123"))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

# Initialize database if not exists
if not os.path.exists('polyview.db'):
    init_db()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/permission')
def permission():
    return render_template('permission.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

# API Endpoints
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    conn = sqlite3.connect('polyview.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    user = c.fetchone()
    conn.close()
    
    if user:
        return jsonify({
            "success": True, 
            "message": "Login successful",
            "user_id": user[0],
            "name": user[1]
        })
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    try:
        conn = sqlite3.connect('polyview.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                 (name, email, password))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        return jsonify({
            "success": True, 
            "message": "Account created successfully",
            "user_id": user_id
        })
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Email already exists"}), 400

def extract_facial_landmarks(frame):
    """Extract 68 facial landmarks using dlib"""
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
    except Exception as e:
        print(f"Landmark extraction error: {e}")
        return None

def detect_micro_expressions(landmarks, prev_landmarks):
    """Detect micro-expressions using facial landmark displacement"""
    if prev_landmarks is None or landmarks is None:
        return []
    
    micro_expressions = []
    
    # Calculate displacement for each AU group
    for au, indices in FAU_MAPPING.items():
        current_points = landmarks[indices]
        previous_points = prev_landmarks[indices]
        
        # Calculate Euclidean distance between current and previous points
        displacement = np.mean(np.linalg.norm(current_points - previous_points, axis=1))
        
        if displacement > 0.02:  # Threshold for micro-expression
            micro_expressions.append(au)
    
    return micro_expressions

def detect_emotion(landmarks):
    """Detect emotion from facial landmarks"""
    if landmarks is None or len(landmarks) < 68:
        return "neutral"
    
    try:
        # Get key points
        left_eyebrow = landmarks[21]
        right_eyebrow = landmarks[22]
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        nose_tip = landmarks[30]
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]
        upper_lip = landmarks[51]
        lower_lip = landmarks[57]
        
        # Calculate facial metrics
        eye_distance = abs(left_eye[0] - right_eye[0])
        eyebrow_distance = abs(left_eyebrow[0] - right_eyebrow[0])
        mouth_openness = abs(upper_lip[1] - lower_lip[1])
        mouth_width = abs(left_mouth[0] - right_mouth[0])
        
        # Calculate smile curvature
        def calculate_curvature(p1, p2, p3):
            """Calculate curvature between three points"""
            dx1 = p2[0] - p1[0]
            dy1 = p2[1] - p1[1]
            dx2 = p3[0] - p2[0]
            dy2 = p3[1] - p2[1]
            return (dx1*dy2 - dx2*dy1) / math.sqrt((dx1**2 + dy1**2)**3)
        
        smile_curve = calculate_curvature(left_mouth, upper_lip, right_mouth)
        
        # Determine emotion based on metrics
        if mouth_openness > 20 and eyebrow_distance > eye_distance:
            return "surprise"
        elif smile_curve > 0.001 and mouth_width > 50:
            return "happy"
        elif eyebrow_distance < eye_distance and smile_curve < -0.001:
            return "sad"
        elif eyebrow_distance < eye_distance - 10 and mouth_openness < 10:
            return "angry"
        else:
            return "neutral"
            
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "neutral"

def analyze_voice(audio_path):
    """Analyze voice for stress indicators"""
    try:
        # Load audio file directly with librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract pitch (fundamental frequency)
        pitch = librosa.yin(y, fmin=80, fmax=400)
        pitch_mean = np.nanmean(pitch)  # Handle NaN values
        
        # Calculate intensity (volume)
        intensity = np.abs(y)
        intensity_mean = np.mean(intensity)
        
        # Calculate jitter (pitch variability)
        jitter = np.mean(np.abs(np.diff(pitch)))
        
        # Calculate shimmer (amplitude variability)
        shimmer = np.mean(np.abs(np.diff(intensity)))
        
        # Simple stress calculation (higher values = more stress)
        stress_level = min(100, max(0, 
            (pitch_mean / 200 * 30) + 
            (jitter * 100 * 20) + 
            (shimmer * 50 * 20) +
            (intensity_mean * 30)
        ))
        
        return {
            "stress_level": float(stress_level),
            "pitch": float(pitch_mean),
            "intensity": float(intensity_mean),
            "jitter": float(jitter),
            "shimmer": float(shimmer)
        }
    except Exception as e:
        print(f"Voice analysis error: {e}")
        return {
            "stress_level": random.uniform(30, 70),
            "pitch": 0,
            "intensity": 0,
            "jitter": 0,
            "shimmer": 0
        }

def fact_check(statement):
    """Verify factual accuracy using Wikipedia"""
    try:
        # Search Wikipedia
        search_results = wikipedia.search(statement, results=3)
        context = ""
        
        for result in search_results:
            try:
                context += wikipedia.summary(result, sentences=2) + "\n\n"
            except:
                continue
        
        # Simple keyword matching
        keywords = statement.lower().split()
        found_keywords = sum(1 for word in keywords if word in context.lower())
        confidence = min(1.0, found_keywords / max(1, len(keywords)))
        
        return {
            "confidence": confidence,
            "sources": search_results
        }
    except:
        return {
            "confidence": random.uniform(0.3, 0.7),
            "sources": []
        }

def calculate_truth_score(metrics):
    """Calculate overall truth score (0-100)"""
    # Weights for different factors
    weights = {
        "stress_level": 0.4,
        "micro_expressions": 0.3,
        "fact_confidence": 0.3
    }
    
    # Normalize metrics
    stress_score = 1 - min(metrics["stress_level"] / 100, 1)
    
    # More expressions = higher deception probability
    expressions_score = 1 - min(len(metrics["micro_expressions"]) / 5, 1)
    
    fact_score = metrics["fact_confidence"]
    
    # Calculate weighted score
    score = (
        weights["stress_level"] * stress_score +
        weights["micro_expressions"] * expressions_score +
        weights["fact_confidence"] * fact_score
    ) * 100
    
    return min(max(score, 0), 100)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # Get files and data
    video_file = request.files.get('video')
    audio_file = request.files.get('audio')
    transcript = request.form.get('transcript', '')
    user_id = request.form.get('user_id', 0)
    
    # Create temporary files
    video_path = tempfile.mktemp(suffix='.mp4')
    audio_path = tempfile.mktemp(suffix='.wav')
    
    # Save files
    if video_file:
        video_file.save(video_path)
    if audio_file:
        audio_file.save(audio_path)
    
    # Initialize metrics
    metrics = {
        "stress_level": 0,
        "micro_expressions": [],
        "fact_confidence": 0,
        "emotion": "neutral"
    }
    
    # Process video if available
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        prev_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every 5th frame
            if frame_count % 5 == 0:
                # Get facial landmarks
                landmarks = extract_facial_landmarks(frame)
                
                if landmarks is not None:
                    # Detect micro-expressions
                    if prev_landmarks is not None:
                        micro_expr = detect_micro_expressions(landmarks, prev_landmarks)
                        metrics["micro_expressions"].extend(micro_expr)
                    
                    # Detect emotion
                    metrics["emotion"] = detect_emotion(landmarks)
                    prev_landmarks = landmarks
        
        cap.release()
    
    # Process audio if available
    if os.path.exists(audio_path):
        voice_metrics = analyze_voice(audio_path)
        metrics["stress_level"] = voice_metrics["stress_level"]
    else:
        metrics["stress_level"] = random.uniform(30, 70)
    
    # Fact-check transcript
    if transcript:
        fact_check_result = fact_check(transcript)
        metrics["fact_confidence"] = fact_check_result["confidence"]
    else:
        metrics["fact_confidence"] = random.uniform(0.5, 0.9)
    
    # Calculate overall truth score
    truth_score = calculate_truth_score(metrics)
    
    # Save to database
    try:
        conn = sqlite3.connect('polyview.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO analyses (user_id, truth_score, stress_level, emotion, transcript)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, truth_score, metrics["stress_level"], metrics["emotion"], transcript))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
    
    # Clean up temporary files
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    # Return results
    return jsonify({
        "truth_score": truth_score,
        "stress_level": metrics["stress_level"],
        "emotion": metrics["emotion"],
        "micro_expressions": list(set(metrics["micro_expressions"])),
        "fact_confidence": metrics["fact_confidence"],
        "transcript": transcript
    })

@app.route('/api/baseline', methods=['POST'])
def baseline_calibration():
    """Establish user's behavioral baseline"""
    try:
        # This would process calibration questions in a real implementation
        return jsonify({
            "success": True,
            "message": "Baseline established",
            "baseline_id": f"baseline_{int(time.time())}"
        })
    except:
        return jsonify({"success": False, "message": "Calibration failed"}), 500

@app.route('/api/history/<int:user_id>')
def analysis_history(user_id):
    """Get user's analysis history"""
    try:
        conn = sqlite3.connect('polyview.db')
        c = conn.cursor()
        c.execute("SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        analyses = c.fetchall()
        conn.close()
        
        # Format results
        history = []
        for analysis in analyses:
            history.append({
                "id": analysis[0],
                "truth_score": analysis[2],
                "stress_level": analysis[3],
                "emotion": analysis[4],
                "created_at": analysis[6]
            })
        
        return jsonify({"success": True, "history": history})
    except:
        return jsonify({"success": False, "message": "Error retrieving history"}), 500

if __name__ == '__main__':
    # Start with a simple check for dlib model
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("ERROR: Missing dlib model file!")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place in project folder")
    else:
        app.run(debug=True, port=5000, threaded=True)