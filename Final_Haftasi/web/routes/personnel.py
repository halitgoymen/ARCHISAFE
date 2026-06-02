import os
import json
import uuid
import pickle
from datetime import datetime, date
from flask import (Blueprint, render_template, redirect, url_for,
                   flash, request, current_app, jsonify)
from flask_login import login_required
from sqlalchemy import func
from werkzeug.utils import secure_filename
from .. import db
from ..models.personnel import Personnel
from ..models.event import AttendanceLog, PPEViolation

personnel_bp = Blueprint('personnel', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

PPE_OPTIONS = [
    ('hardhat', 'Baret'),
    ('vest', 'Reflektörlü Yelek'),
    ('mask', 'Koruyucu Maske'),
    ('gloves', 'Eldiven'),
    ('shoes', 'İş Ayakkabısı / Botu'),
    ('Ear-protection', 'Kulak Koruyucu'),
]


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _next_personnel_id():
    last = Personnel.query.order_by(Personnel.id.desc()).first()
    num = (last.id + 1) if last else 1
    return f'EMP{num:04d}'


@personnel_bp.route('/')
@login_required
def index():
    search = request.args.get('q', '').strip()
    area = request.args.get('area', '')
    status = request.args.get('status', '')

    query = Personnel.query
    if search:
        like = f'%{search}%'
        query = query.filter(
            db.or_(
                Personnel.first_name.ilike(like),
                Personnel.last_name.ilike(like),
                Personnel.personnel_id.ilike(like),
                Personnel.job_title.ilike(like),
            )
        )
    if area:
        query = query.filter(Personnel.work_area == area)
    if status == 'active':
        query = query.filter_by(is_active=True)
    elif status == 'onsite':
        query = query.filter_by(is_on_site=True)
    elif status == 'inactive':
        query = query.filter_by(is_active=False)

    personnel_list = query.order_by(Personnel.last_name).all()
    areas = db.session.query(Personnel.work_area).distinct().filter(
        Personnel.work_area.isnot(None)
    ).all()
    areas = [a[0] for a in areas]

    return render_template(
        'personnel/index.html',
        personnel_list=personnel_list,
        areas=areas,
        search=search,
        selected_area=area,
        selected_status=status,
        ppe_options=PPE_OPTIONS,
    )


@personnel_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add():
    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        job_title = request.form.get('job_title', '').strip()
        work_area = request.form.get('work_area', '').strip()
        phone = request.form.get('phone', '').strip()
        email = request.form.get('email', '').strip()
        required_ppe = request.form.getlist('required_ppe')

        if not first_name or not last_name:
            flash('Ad ve soyad zorunludur.', 'danger')
            return redirect(url_for('personnel.add'))

        p = Personnel(
            personnel_id=_next_personnel_id(),
            first_name=first_name,
            last_name=last_name,
            job_title=job_title,
            work_area=work_area,
            phone=phone,
            email=email,
        )
        p.required_ppe = required_ppe

        photo = request.files.get('photo')
        if photo and _allowed_file(photo.filename):
            ext = photo.filename.rsplit('.', 1)[1].lower()
            filename = f'{uuid.uuid4().hex}.{ext}'
            photo_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'personnel')
            photo.save(os.path.join(photo_dir, filename))
            p.photo_path = f'uploads/personnel/{filename}'

            # Try to generate face encoding
            try:
                import face_recognition
                import numpy as np
                import cv2
                img_path = os.path.join(photo_dir, filename)
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    p.face_encoding = pickle.dumps(encodings[0])
            except Exception:
                pass

        db.session.add(p)
        db.session.commit()
        flash(f'Personel {p.full_name} ({p.personnel_id}) başarıyla eklendi.', 'success')
        return redirect(url_for('personnel.detail', pid=p.id))

    return render_template('personnel/add.html', ppe_options=PPE_OPTIONS,
                           next_id=_next_personnel_id())


@personnel_bp.route('/<int:pid>')
@login_required
def detail(pid):
    p = Personnel.query.get_or_404(pid)
    today = date.today()

    recent_logs = (
        AttendanceLog.query.filter_by(personnel_id=pid)
        .order_by(AttendanceLog.entry_time.desc())
        .limit(15).all()
    )
    recent_violations = (
        PPEViolation.query.filter_by(personnel_id=pid)
        .order_by(PPEViolation.timestamp.desc())
        .limit(10).all()
    )
    today_log = AttendanceLog.query.filter(
        AttendanceLog.personnel_id == pid,
        func.date(AttendanceLog.entry_time) == today
    ).first()

    ppe_stats = {
        'total_violations': PPEViolation.query.filter_by(personnel_id=pid).count(),
        'unresolved': PPEViolation.query.filter_by(personnel_id=pid, is_resolved=False).count(),
        'total_days': AttendanceLog.query.filter_by(personnel_id=pid).distinct(
            func.date(AttendanceLog.entry_time)
        ).count(),
    }

    return render_template(
        'personnel/detail.html',
        p=p,
        recent_logs=recent_logs,
        recent_violations=recent_violations,
        today_log=today_log,
        ppe_stats=ppe_stats,
        ppe_options=PPE_OPTIONS,
    )


@personnel_bp.route('/<int:pid>/edit', methods=['POST'])
@login_required
def edit(pid):
    p = Personnel.query.get_or_404(pid)
    p.first_name = request.form.get('first_name', p.first_name).strip()
    p.last_name = request.form.get('last_name', p.last_name).strip()
    p.job_title = request.form.get('job_title', p.job_title)
    p.work_area = request.form.get('work_area', p.work_area)
    p.phone = request.form.get('phone', p.phone)
    p.email = request.form.get('email', p.email)
    p.required_ppe = request.form.getlist('required_ppe')
    p.updated_at = datetime.utcnow()

    photo = request.files.get('photo')
    if photo and _allowed_file(photo.filename):
        ext = photo.filename.rsplit('.', 1)[1].lower()
        filename = f'{uuid.uuid4().hex}.{ext}'
        photo_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'personnel')
        photo.save(os.path.join(photo_dir, filename))
        p.photo_path = f'uploads/personnel/{filename}'

        try:
            import face_recognition
            img = face_recognition.load_image_file(os.path.join(photo_dir, filename))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                p.face_encoding = pickle.dumps(encodings[0])
        except Exception:
            pass

    db.session.commit()
    flash('Personel bilgileri güncellendi.', 'success')
    return redirect(url_for('personnel.detail', pid=pid))


@personnel_bp.route('/<int:pid>/toggle-active', methods=['POST'])
@login_required
def toggle_active(pid):
    p = Personnel.query.get_or_404(pid)
    p.is_active = not p.is_active
    if not p.is_active:
        p.is_on_site = False
    db.session.commit()
    status = 'aktif' if p.is_active else 'pasif'
    flash(f'{p.full_name} {status} olarak işaretlendi.', 'info')
    return redirect(url_for('personnel.detail', pid=pid))


@personnel_bp.route('/<int:pid>/checkout', methods=['POST'])
@login_required
def checkout(pid):
    p = Personnel.query.get_or_404(pid)
    log = AttendanceLog.query.filter_by(
        personnel_id=pid, exit_time=None
    ).order_by(AttendanceLog.entry_time.desc()).first()

    if log:
        log.exit_time = datetime.utcnow()
    p.is_on_site = False
    db.session.commit()
    flash(f'{p.full_name} sahasından çıkış yaptı.', 'info')
    return redirect(url_for('personnel.detail', pid=pid))


@personnel_bp.route('/api/search')
@login_required
def api_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    like = f'%{q}%'
    results = Personnel.query.filter(
        db.or_(
            Personnel.first_name.ilike(like),
            Personnel.last_name.ilike(like),
            Personnel.personnel_id.ilike(like),
        )
    ).limit(10).all()
    return jsonify([{
        'id': p.id,
        'personnel_id': p.personnel_id,
        'full_name': p.full_name,
        'job_title': p.job_title,
        'photo_path': p.photo_path,
    } for p in results])
