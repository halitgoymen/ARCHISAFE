from datetime import datetime, date, timedelta
from flask import Blueprint, render_template, jsonify
from flask_login import login_required
from sqlalchemy import func
from .. import db
from ..models.personnel import Personnel
from ..models.camera import Camera
from ..models.event import AttendanceLog, PPEViolation, FallEvent, Alert

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
@login_required
def index():
    today = date.today()
    now = datetime.utcnow()

    stats = {
        'active_personnel': Personnel.query.filter_by(is_on_site=True, is_active=True).count(),
        'today_entries': AttendanceLog.query.filter(
            func.date(AttendanceLog.entry_time) == today
        ).count(),
        'ppe_warnings': PPEViolation.query.filter_by(is_resolved=False).count(),
        'fall_events_today': FallEvent.query.filter(
            func.date(FallEvent.timestamp) == today
        ).count(),
        'active_cameras': Camera.query.filter_by(is_active=True).count(),
        'total_cameras': Camera.query.count(),
        'unread_alerts': Alert.query.filter_by(is_read=False).count(),
    }

    recent_entries = (
        AttendanceLog.query
        .filter(func.date(AttendanceLog.entry_time) == today)
        .order_by(AttendanceLog.entry_time.desc())
        .limit(10).all()
    )

    recent_violations = (
        PPEViolation.query
        .filter_by(is_resolved=False)
        .order_by(PPEViolation.timestamp.desc())
        .limit(8).all()
    )

    recent_falls = (
        FallEvent.query
        .order_by(FallEvent.timestamp.desc())
        .limit(5).all()
    )

    alerts = (
        Alert.query
        .filter_by(is_read=False)
        .order_by(Alert.timestamp.desc())
        .limit(10).all()
    )

    cameras = Camera.query.filter_by(is_active=True).all()

    # Weekly entry chart data (last 7 days)
    week_labels = []
    week_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = AttendanceLog.query.filter(
            func.date(AttendanceLog.entry_time) == day
        ).count()
        week_labels.append(day.strftime('%d/%m'))
        week_data.append(count)

    # PPE violation breakdown
    ppe_breakdown = db.session.query(
        PPEViolation._missing_ppe,
        func.count(PPEViolation.id)
    ).group_by(PPEViolation._missing_ppe).all()

    return render_template(
        'dashboard/index.html',
        stats=stats,
        recent_entries=recent_entries,
        recent_violations=recent_violations,
        recent_falls=recent_falls,
        alerts=alerts,
        cameras=cameras,
        week_labels=week_labels,
        week_data=week_data,
    )


@dashboard_bp.route('/api/stats')
@login_required
def api_stats():
    today = date.today()
    stats = {
        'active_personnel': Personnel.query.filter_by(is_on_site=True, is_active=True).count(),
        'today_entries': AttendanceLog.query.filter(
            func.date(AttendanceLog.entry_time) == today
        ).count(),
        'ppe_warnings': PPEViolation.query.filter_by(is_resolved=False).count(),
        'fall_events_today': FallEvent.query.filter(
            func.date(FallEvent.timestamp) == today
        ).count(),
        'unread_alerts': Alert.query.filter_by(is_read=False).count(),
    }
    return jsonify(stats)
