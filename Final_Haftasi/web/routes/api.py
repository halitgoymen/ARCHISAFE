"""REST API endpoints for AI services and external integrations."""
from datetime import datetime
from flask import Blueprint, jsonify, request
from flask_login import login_required
from .. import db
from ..models.personnel import Personnel
from ..models.camera import Camera
from ..models.event import AttendanceLog, PPEViolation, FallEvent, Alert

api_bp = Blueprint('api', __name__)


@api_bp.route('/alerts/unread')
@login_required
def unread_alerts():
    alerts = (
        Alert.query.filter_by(is_read=False)
        .order_by(Alert.timestamp.desc())
        .limit(20).all()
    )
    return jsonify([{
        'id': a.id,
        'type': a.alert_type,
        'severity': a.severity,
        'title': a.title,
        'message': a.message,
        'time_ago': a.time_ago,
        'timestamp': a.timestamp.isoformat(),
    } for a in alerts])


@api_bp.route('/alerts/<int:aid>/read', methods=['POST'])
@login_required
def mark_alert_read(aid):
    alert = Alert.query.get_or_404(aid)
    alert.is_read = True
    db.session.commit()
    return jsonify({'success': True})


@api_bp.route('/alerts/read-all', methods=['POST'])
@login_required
def read_all_alerts():
    Alert.query.filter_by(is_read=False).update({'is_read': True})
    db.session.commit()
    return jsonify({'success': True})


@api_bp.route('/face-checkin', methods=['POST'])
def face_checkin():
    """Called by face recognition service when a person is detected at entry."""
    data = request.json or {}
    personnel_id = data.get('personnel_id')
    camera_id = data.get('camera_id')
    confidence = data.get('confidence', 0.0)

    p = Personnel.query.filter_by(personnel_id=personnel_id).first()
    if not p:
        return jsonify({'success': False, 'error': 'Personnel not found'}), 404

    cam = Camera.query.filter_by(camera_id=camera_id).first() if camera_id else None

    # Prevent duplicate entries within 1 hour
    from datetime import timedelta
    recent = AttendanceLog.query.filter(
        AttendanceLog.personnel_id == p.id,
        AttendanceLog.entry_time >= datetime.utcnow() - timedelta(hours=1),
        AttendanceLog.exit_time.is_(None)
    ).first()

    if recent:
        return jsonify({'success': False, 'error': 'Already checked in', 'personnel': p.full_name})

    log = AttendanceLog(
        personnel_id=p.id,
        camera_id=cam.id if cam else None,
        entry_time=datetime.utcnow(),
        date=datetime.utcnow().date(),
        face_confidence=confidence,
        method='face',
    )
    p.is_on_site = True
    db.session.add(log)
    db.session.commit()

    from .. import socketio
    socketio.emit('personnel_checkin', {
        'personnel_id': p.personnel_id,
        'full_name': p.full_name,
        'job_title': p.job_title,
        'photo_path': p.photo_path,
        'entry_time': log.entry_time.isoformat(),
        'required_ppe': p.required_ppe,
    })

    return jsonify({
        'success': True,
        'personnel': p.full_name,
        'entry_time': log.entry_time.isoformat(),
        'required_ppe': p.required_ppe,
    })


@api_bp.route('/ppe-violation', methods=['POST'])
def log_ppe_violation():
    """Called by YOLO PPE detection service."""
    data = request.json or {}
    personnel_id = data.get('personnel_id')
    camera_id = data.get('camera_id')

    p = Personnel.query.filter_by(personnel_id=personnel_id).first() if personnel_id else None
    cam = Camera.query.filter_by(camera_id=camera_id).first() if camera_id else None

    violation = PPEViolation(
        personnel_id=p.id if p else None,
        camera_id=cam.id if cam else None,
        missing_ppe=data.get('missing_ppe', []),
        detected_ppe=data.get('detected_ppe', []),
        confidence=data.get('confidence', 0.0),
        image_path=data.get('image_path', ''),
    )
    db.session.add(violation)
    db.session.flush()

    if violation.missing_ppe:
        alert = Alert(
            alert_type='ppe_violation',
            severity='high' if len(violation.missing_ppe) >= 2 else 'medium',
            title='KKD İhlali Tespit Edildi',
            message=f'{p.full_name if p else "Bilinmeyen"} — Eksik: {", ".join(violation.missing_ppe)}',
            reference_id=violation.id,
            reference_type='ppe_violation',
            camera_id=cam.id if cam else None,
            personnel_id=p.id if p else None,
        )
        db.session.add(alert)

    db.session.commit()

    from .. import socketio
    socketio.emit('ppe_violation', {
        'personnel': p.full_name if p else 'Bilinmeyen',
        'missing_ppe': violation.missing_ppe,
        'camera': cam.name if cam else camera_id,
    })

    return jsonify({'success': True, 'violation_id': violation.id})


@api_bp.route('/cameras')
@login_required
def get_cameras():
    cameras = Camera.query.all()
    return jsonify([{
        'id': c.id,
        'camera_id': c.camera_id,
        'name': c.name,
        'type': c.camera_type,
        'is_active': c.is_active,
        'stream_url': c.stream_url,
    } for c in cameras])


@api_bp.route('/personnel/onsite')
@login_required
def onsite_personnel():
    personnel = Personnel.query.filter_by(is_on_site=True, is_active=True).all()
    return jsonify([{
        'id': p.id,
        'personnel_id': p.personnel_id,
        'full_name': p.full_name,
        'job_title': p.job_title,
        'required_ppe': p.required_ppe,
    } for p in personnel])
