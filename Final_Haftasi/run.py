"""ARCHISAFE Web Application Entry Point
Usage:
    python run.py
    -- or --
    python -m web.run
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web import create_app, socketio

app = create_app('development')

if __name__ == '__main__':
    print('=' * 60)
    print('  ARCHISAFE — İnşaat Güvenlik Yönetim Sistemi')
    print('  http://localhost:5000')
    print('  Varsayılan: admin / admin123')
    print('=' * 60)
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
