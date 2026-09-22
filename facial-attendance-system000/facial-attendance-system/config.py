import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "attendance.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    STUDENT_PHOTOS_DIR = os.path.join(BASE_DIR, "static", "uploads", "students")
    CLASSROOM_PHOTOS_DIR = os.path.join(BASE_DIR, "static", "uploads", "classroom")
    MODELS_DIR = os.path.join(BASE_DIR, "static", "models")

    # Minimum photos required per student before the recognizer can be trained on them
    MIN_PHOTOS_PER_STUDENT = 1
    # FaceNet embedding-distance threshold (LOWER is a better match).
    # 0.9 is a reasonable default for keras-facenet's L2-normalized 512-d
    # embeddings. Lower it (e.g. 0.75) to be stricter and cut false positives;
    # raise it if too many known faces are coming back "Unknown".
    RECOGNITION_CONFIDENCE_THRESHOLD = 0.15

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB uploads
