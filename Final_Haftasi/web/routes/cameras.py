import json
from datetime import datetime
from flask import (Blueprint, render_template, redirect, url_for,
                   flash, request, jsonify, Response, current_app)
from flask_login import login_required
from .. import db
from ..models.camera import Camera
from ..models.event import Alert

cameras_bp = Blueprint('cameras', __name__)


@cameras_bp.route('/')
@login_required
def index():
    cameras = Camera.query.order_by(Camera.camera_type, Camera.name).all()
    return render_template('cameras/index.html', cameras=cameras)


@cameras_bp.route('/add', methods=['POST'])
@login_required
def add():
    name = request.form.get('name', '').strip()
    location = request.form.get('location', '').strip()
    stream_url = request.form.get('stream_url', '').strip()
    camera_type = request.form.get('camera_type', 'general')
    resolution = request.form.get('resolution', '1280x720')
    fps = int(request.form.get('fps', 25))
    notes = request.form.get('notes', '')

    if not name or not stream_url:
        flash('Kamera adı ve akış adresi zorunludur.', 'danger')
        return redirect(url_for('cameras.index'))

    last = Camera.query.order_by(Camera.id.desc()).first()
    num = (last.id + 1) if last else 1
    camera_id = f'CAM-{num:03d}'
    while Camera.query.filter_by(camera_id=camera_id).first():
        num += 1
        camera_id = f'CAM-{num:03d}'

    cam = Camera(
        camera_id=camera_id,
        name=name,
        location=location,
        stream_url=stream_url,
        camera_type=camera_type,
        resolution=resolution,
        fps=fps,
        notes=notes,
        is_active=True,
    )
    db.session.add(cam)
    db.session.commit()
    flash(f'Kamera {camera_id} ({name}) eklendi.', 'success')
    return redirect(url_for('cameras.index'))


@cameras_bp.route('/<int:cid>/edit', methods=['POST'])
@login_required
def edit(cid):
    cam = Camera.query.get_or_404(cid)
    cam.name = request.form.get('name', cam.name).strip()
    cam.location = request.form.get('location', cam.location)
    cam.stream_url = request.form.get('stream_url', cam.stream_url)
    cam.camera_type = request.form.get('camera_type', cam.camera_type)
    cam.resolution = request.form.get('resolution', cam.resolution)
    cam.fps = int(request.form.get('fps', cam.fps))
    cam.notes = request.form.get('notes', cam.notes)
    db.session.commit()
    flash(f'Kamera {cam.camera_id} güncellendi.', 'success')
    return redirect(url_for('cameras.index'))


@cameras_bp.route('/<int:cid>/toggle', methods=['POST'])
@login_required
def toggle(cid):
    cam = Camera.query.get_or_404(cid)
    cam.is_active = not cam.is_active
    db.session.commit()
    status = 'aktif' if cam.is_active else 'pasif'
    return jsonify({'success': True, 'is_active': cam.is_active, 'status': status})


@cameras_bp.route('/<int:cid>/delete', methods=['POST'])
@login_required
def delete(cid):
    cam = Camera.query.get_or_404(cid)
    name = cam.name
    db.session.delete(cam)
    db.session.commit()
    flash(f'Kamera {name} silindi.', 'warning')
    return redirect(url_for('cameras.index'))


@cameras_bp.route('/<int:cid>/roi', methods=['POST'])
@login_required
def save_roi(cid):
    cam = Camera.query.get_or_404(cid)
    roi_data = request.json.get('roi', [])
    cam.roi_data = roi_data
    db.session.commit()
    return jsonify({'success': True})


@cameras_bp.route('/<int:cid>/live')
@login_required
def live(cid):
    cam = Camera.query.get_or_404(cid)
    return render_template('cameras/live.html', camera=cam)


@cameras_bp.route('/<int:cid>/stream')
@login_required
def stream(cid):
    cam = Camera.query.get_or_404(cid)
    from ..services.camera_service import CameraService
    return Response(
        CameraService.generate_frames(cam),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
