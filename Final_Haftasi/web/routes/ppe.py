from datetime import datetime, date
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required
from sqlalchemy import func
from .. import db
from ..models.personnel import Personnel
from ..models.camera import Camera
from ..models.event import PPEViolation, Alert

ppe_bp = Blueprint('ppe', __name__)


@ppe_bp.route('/')
@login_required
def index():
    page = request.args.get('page', 1, type=int)
    filter_resolved = request.args.get('resolved', '0')
    filter_personnel = request.args.get('personnel_id', '', type=str)
    filter_date = request.args.get('date', '')

    query = PPEViolation.query
    if filter_resolved == '0':
        query = query.filter_by(is_resolved=False)
    elif filter_resolved == '1':
        query = query.filter_by(is_resolved=True)

    if filter_personnel:
        p = Personnel.query.filter_by(personnel_id=filter_personnel).first()
        if p:
            query = query.filter_by(personnel_id=p.id)

    if filter_date:
        try:
            d = datetime.strptime(filter_date, '%Y-%m-%d').date()
            query = query.filter(func.date(PPEViolation.timestamp) == d)
        except ValueError:
            pass

    violations = query.order_by(PPEViolation.timestamp.desc()).paginate(
        page=page, per_page=20, error_out=False
    )

    today = date.today()
    stats = {
        'total_unresolved': PPEViolation.query.filter_by(is_resolved=False).count(),
        'today_count': PPEViolation.query.filter(
            func.date(PPEViolation.timestamp) == today
        ).count(),
        'personnel_with_violations': db.session.query(
            func.count(func.distinct(PPEViolation.personnel_id))
        ).filter_by(is_resolved=False).scalar() or 0,
    }

    onsite_personnel = Personnel.query.filter_by(is_on_site=True, is_active=True).all()
    cameras = Camera.query.filter_by(is_active=True).all()

    return render_template(
        'ppe/index.html',
        violations=violations,
        stats=stats,
        onsite_personnel=onsite_personnel,
        cameras=cameras,
        filter_resolved=filter_resolved,
        filter_personnel=filter_personnel,
        filter_date=filter_date,
    )


@ppe_bp.route('/<int:vid>/resolve', methods=['POST'])
@login_required
def resolve(vid):
    v = PPEViolation.query.get_or_404(vid)
    v.is_resolved = True
    v.resolved_at = datetime.utcnow()
    v.notes = request.form.get('notes', '')
    db.session.commit()
    flash('İhlal çözüldü olarak işaretlendi.', 'success')
    return redirect(request.referrer or url_for('ppe.index'))


@ppe_bp.route('/resolve-all', methods=['POST'])
@login_required
def resolve_all():
    PPEViolation.query.filter_by(is_resolved=False).update({
        'is_resolved': True,
        'resolved_at': datetime.utcnow()
    })
    db.session.commit()
    flash('Tüm ihlaller çözüldü olarak işaretlendi.', 'success')
    return redirect(url_for('ppe.index'))


@ppe_bp.route('/api/live-status')
@login_required
def api_live_status():
    onsite = Personnel.query.filter_by(is_on_site=True, is_active=True).all()
    results = []
    for p in onsite:
        latest_v = PPEViolation.query.filter_by(
            personnel_id=p.id, is_resolved=False
        ).order_by(PPEViolation.timestamp.desc()).first()

        results.append({
            'id': p.id,
            'personnel_id': p.personnel_id,
            'full_name': p.full_name,
            'job_title': p.job_title,
            'photo_path': p.photo_path,
            'required_ppe': p.required_ppe,
            'missing_ppe': latest_v.missing_ppe if latest_v else [],
            'detected_ppe': latest_v.detected_ppe if latest_v else p.required_ppe,
            'has_violation': latest_v is not None,
            'violation_id': latest_v.id if latest_v else None,
        })

    return jsonify(results)
