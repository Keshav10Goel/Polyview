from app import create_app
from app.models import db, User, Analysis

app = create_app()

@app.before_first_request
def create_tables():
    db.create_all()
    
    # Create a test user if none exists
    if not User.query.first():
        test_user = User(
            name="Test User",
            email="test@example.com",
            password="password"  # In real app, hash this
        )
        db.session.add(test_user)
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, port=5000)