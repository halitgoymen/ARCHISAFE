import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_socketio import SocketIO

db = SQLAlchemy()
login_manager = LoginManager()
socketio = SocketIO()


def create_app(config_name='default'):
    app = Flask(__name__)

    from .config import config
    app.config.from_object(config[config_name])

    db.init_app(app)
    login_manager.init_app(app)
    socketio.init_app(app, cors_allowed_origins='*', async_mode='threading')

    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Bu sayfaya erişmek için giriş yapmanız gerekiyor.'
    login_manager.login_message_category = 'warning'

    from .models.user import User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Blueprints
    from .routes.auth import auth_bp
    from .routes.dashboard import dashboard_bp
    from .routes.personnel import personnel_bp
    from .routes.cameras import cameras_bp
    from .routes.ppe import ppe_bp
    from .routes.fall_detection import fall_bp
    from .routes.reports import reports_bp
    from .routes.api import api_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(personnel_bp, url_prefix='/personnel')
    app.register_blueprint(cameras_bp, url_prefix='/cameras')
    app.register_blueprint(ppe_bp, url_prefix='/ppe')
    app.register_blueprint(fall_bp, url_prefix='/fall-detection')
    app.register_blueprint(reports_bp, url_prefix='/reports')
    app.register_blueprint(api_bp, url_prefix='/api')

    upload_base = app.config['UPLOAD_FOLDER']
    for sub in ('personnel', 'events', 'fall', 'ppe'):
        os.makedirs(os.path.join(upload_base, sub), exist_ok=True)

    with app.app_context():
        db.create_all()
        _seed_default_data()

    return app


def _seed_default_data():
    from .models.user import User
    from .models.camera import Camera

    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            full_name='Sistem Yöneticisi',
            email='admin@archisafe.com',
            role='admin',
        )
        admin.set_password('admin123')
        db.session.add(admin)

    if not Camera.query.first():
        demo_cameras = [
            Camera(camera_id='CAM-001', name='Giriş Kapısı', location='Ana Giriş',
                   stream_url='0', camera_type='entry', is_active=True),
            Camera(camera_id='CAM-002', name='A Blok KKD Kontrol', location='A Blok Zemin',
                   stream_url='1', camera_type='ppe', is_active=True),
            Camera(camera_id='CAM-003', name='İskele İzleme', location='B Blok 3. Kat',
                   stream_url='2', camera_type='fall_detection', is_active=False),
            Camera(camera_id='CAM-004', name='Depo Alanı', location='Depo Binası',
                   stream_url='3', camera_type='general', is_active=False),
        ]
        db.session.bulk_save_objects(demo_cameras)

    db.session.commit()
