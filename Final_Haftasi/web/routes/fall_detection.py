from datetime import datetime, date
from flask import (Blueprint, render_template, request, jsonify,
                   redirect, url_for, flash)
from flask_login import login_required, current_user
from sqlalchemy import func
from .. import db
from ..models.camera import Camera
from ..models.event import FallEvent, Alert

fall_bp = Blueprint('fall', __name__)


@fall_bp.route('/')
@login_required
def index():
    page = request.args.get('page', 1, type=int)
    filter_resolved = request.args.get('resolved', '0')
    filter_camera = request.args.get('camera_id', '', type=str)
    filter_date = request.args.get('date', '')

    query = FallEvent.query
    if filter_resolved == '0':
        query = query.filter_by(is_resolved=False)
    elif filter_resolved == '1':
        query = query.filter_by(is_resolved=True)

    if filter_camera:
        cam = Camera.query.filter_by(camera_id=filter_camera).first()
        if cam:
            query = query.filter_by(camera_id=cam.id)

    if filter_date:
        try:
            d = datetime.strptime(filter_date, '%Y-%m-%d').date()
            query = query.filter(func.date(FallEvent.timestamp) == d)
        except ValueError:
            pass

    events = query.order_by(FallEvent.timestamp.desc()).paginate(
        page=page, per_page=15, error_out=False
    )

    today = date.today()
    stats = {
        'total_unresolved': FallEvent.query.filter_by(is_resolved=False).count(),
        'today_count': FallEvent.query.filter(
            func.date(FallEvent.timestamp) == today
        ).count(),
        'this_week': FallEvent.query.filter(
            FallEvent.timestamp >= datetime.utcnow().replace(
                hour=0, minute=0, second=0
            ) - __import__('datetime').timedelta(days=7)
        ).count(),
    }

    cameras = Camera.query.filter(
        db.or_(
            Camera.camera_type == 'fall_detection',
            Camera.camera_type == 'general'
        )
    ).all()

    return render_template(
        'fall_detection/index.html',
        events=events,
        stats=stats,
        cameras=cameras,
        filter_resolved=filter_resolved,
        filter_camera=filter_camera,
        filter_date=filter_date,
    )


@fall_bp.route('/<int:eid>/resolve', methods=['POST'])
@login_required
def resolve(eid):
    event = FallEvent.query.get_or_404(eid)
    event.is_resolved = True
    event.resolved_at = datetime.utcnow()
    event.resolved_by = current_user.id
    event.notes = request.form.get('notes', '')

    unread_alert = Alert.query.filter_by(
        reference_id=eid, reference_type='fall_event', is_resolved=False
    ).first()
    if unread_alert:
        unread_alert.is_resolved = True
        unread_alert.is_read = True

    db.session.commit()
    flash(f'Düşme olayı #{eid} çözüldü olarak işaretlendi.', 'success')
    return redirect(request.referrer or url_for('fall.index'))


@fall_bp.route('/api/recent')
@login_required
def api_recent():
    events = (
        FallEvent.query
        .filter_by(is_resolved=False)
        .order_by(FallEvent.timestamp.desc())
        .limit(10).all()
    )
    return jsonify([{
        'id': e.id,
        'camera_id': e.camera.camera_id if e.camera else None,
        'camera_name': e.camera.name if e.camera else 'Bilinmiyor',
        'timestamp': e.timestamp.isoformat(),
        'confidence': e.confidence,
        'image_path': e.image_path,
        'location': e.location_desc,
    } for e in events])


@fall_bp.route('/api/create', methods=['POST'])
def api_create():
    """Called by AI detection service to log a fall event."""
    data = request.json or {}
    camera_id = data.get('camera_id')
    cam = Camera.query.filter_by(camera_id=camera_id).first() if camera_id else None

    event = FallEvent(
        camera_id=cam.id if cam else None,
        confidence=data.get('confidence', 0.0),
        location_desc=data.get('location', ''),
        image_path=data.get('image_path', ''),
        video_path=data.get('video_path', ''),
    )
    db.session.add(event)
    db.session.flush()

    alert = Alert(
        alert_type='fall',
        severity='critical',
        title='Düşme Tespit Edildi!',
        message=f'Kamera: {cam.name if cam else camera_id} — Güven: {event.confidence:.0%}',
        reference_id=event.id,
        reference_type='fall_event',
        camera_id=cam.id if cam else None,
    )
    db.session.add(alert)
    db.session.commit()

    from .. import socketio
    socketio.emit('fall_alert', {
        'event_id': event.id,
        'camera_name': cam.name if cam else camera_id,
        'confidence': event.confidence,
        'timestamp': event.timestamp.isoformat(),
    })

    return jsonify({'success': True, 'event_id': event.id})
