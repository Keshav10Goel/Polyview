from flask import Blueprint, render_template, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from .models import User, Analysis
from .utils.video_processor import process_video, save_uploaded_file
from .utils.audio_analyzer import analyze_audio
import os

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/permission')
def permission():
    return render_template('permission.html')

@bp.route('/analysis')
@login_required
def analysis_page():
    return render_template('analysis.html')

@bp.route('/features')
def features():
    return render_template('features.html')

@bp.route('/pricing')
def pricing():
    return render_template('pricing.html')

@bp.route('/contact')
def contact():
    return render_template('contact.html')

@bp.route('/login')
def login_page():
    return render_template('login.html')

@bp.route('/signup')
def signup_page():
    return render_template('signup.html')

@bp.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if user and check_password_hash(user.password, password):
        login_user(user)
        return jsonify({
            "success": True, 
            "message": "Login successful",
            "user_id": user.id,
            "name": user.name
        })
    return jsonify({"success": False, "message": "Invalid credentials"}), 401

@bp.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email exists"}), 400
    
    hashed_password = generate_password_hash(password)
    new_user = User(name=name, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({
        "success": True, 
        "message": "Account created",
        "user_id": new_user.id
    })

@bp.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    if 'video' in request.files:
        video_file = request.files['video']
        upload_folder = 'uploads'
        video_path = save_uploaded_file(video_file, upload_folder)
        
        # Simulated results
        return jsonify({
            "truth_score": 85.0,
            "stress_level": 30.0,
            "emotion": "neutral",
            "micro_expressions": ["AU1", "AU4"],
            "transcript": "Sample transcript text"
        })
    return jsonify({"error": "No video file"}), 400

@bp.route('/api/history', methods=['GET'])
@login_required
def analysis_history():
    analyses = Analysis.query.filter_by(user_id=current_user.id).all()
    history = [{
        "id": a.id,
        "truth_score": a.truth_score,
        "created_at": a.created_at.isoformat()
    } for a in analyses]
    return jsonify({"history": history})