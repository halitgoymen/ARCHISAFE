import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'archisafe-secret-key-2024-ultra-secure')
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'archisafe.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB

    # YOLO model paths (relative to project root)
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    YOLO_MODEL_V6 = os.path.join(PROJECT_ROOT, 'Hafta-6', 'results', 'ARCHISAFE_v6',
                                  'yolo12n_150ep', 'weights', 'best.pt')
    YOLO_MODEL_V5 = os.path.join(PROJECT_ROOT, 'Hafta-5', 'results', 'ARCHISAFE_v5',
                                  'yolo26n_200ep', 'weights', 'best.pt')
    YOLO_MODEL_V4 = os.path.join(PROJECT_ROOT, 'Hafta-4', 'results', 'ARCHISAFE_Final',
                                  'YOLO12s_50ep', 'weights', 'best.pt')

    YOLO_CONFIDENCE = 0.35
    YOLO_IOU = 0.45
    YOLO_IMG_SIZE = 640

    # Face recognition
    FACE_RECOGNITION_TOLERANCE = 0.55
    FACE_ENCODINGS_PATH = os.path.join(BASE_DIR, 'static', 'uploads', 'encodings.pkl')

    # Camera
    CAMERA_FRAME_WIDTH = 1280
    CAMERA_FRAME_HEIGHT = 720
    CAMERA_FPS = 25

    # Alerts
    FALL_CONSECUTIVE_FRAMES = 3
    PPE_CHECK_INTERVAL = 5  # seconds

    # Class names from YOLO model
    CLASS_NAMES = [
        'Ear-protection', 'Fall Detection', 'gloves', 'hardhat',
        'mask', 'no_arm_sleeve', 'person', 'shoes',
        'ungloves', 'unhardhat', 'unmask', 'unvest', 'vest'
    ]
    VIOLATION_CLASSES = {'ungloves', 'unhardhat', 'unmask', 'unvest', 'no_arm_sleeve'}
    SAFE_PPE_CLASSES = {'gloves', 'hardhat', 'mask', 'vest', 'shoes', 'Ear-protection'}


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
