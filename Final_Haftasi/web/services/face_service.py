"""Face recognition service for personnel check-in."""
import os
import pickle
import threading
import numpy as np

_known_encodings = {}  # personnel_id -> np.array
_encodings_lock = threading.Lock()
_initialized = False


def load_encodings():
    """Load all face encodings from the database into memory."""
    global _initialized
    with _encodings_lock:
        try:
            from web import db
            from web.models.personnel import Personnel
            personnel = Personnel.query.filter(
                Personnel.face_encoding.isnot(None),
                Personnel.is_active == True
            ).all()
            _known_encodings.clear()
            for p in personnel:
                try:
                    enc = pickle.loads(p.face_encoding)
                    _known_encodings[p.personnel_id] = enc
                except Exception:
                    pass
            _initialized = True
            print(f'[Face] {len(_known_encodings)} yüz kodlaması yüklendi.')
        except Exception as e:
            print(f'[Face] Kodlamalar yüklenemedi: {e}')


def identify_face(face_encoding, tolerance=0.55):
    """Match a face encoding against known personnel.
    Returns (personnel_id, confidence) or (None, 0.0).
    """
    if not _initialized:
        load_encodings()

    if not _known_encodings:
        return None, 0.0

    try:
        import face_recognition
        known_ids = list(_known_encodings.keys())
        known_encs = np.array(list(_known_encodings.values()))

        distances = face_recognition.face_distance(known_encs, face_encoding)
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]

        if best_dist <= tolerance:
            confidence = 1.0 - best_dist
            return known_ids[best_idx], confidence
        return None, 0.0
    except Exception as e:
        print(f'[Face] Identification error: {e}')
        return None, 0.0


def detect_and_identify(frame, tolerance=0.55):
    """Detect all faces in a frame and identify them.
    Returns list of dicts: {location, personnel_id, confidence, encoding}
    """
    try:
        import face_recognition
        rgb = frame[:, :, ::-1]  # BGR to RGB
        locations = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, locations)

        results = []
        for loc, enc in zip(locations, encodings):
            pid, conf = identify_face(enc, tolerance)
            results.append({
                'location': loc,  # (top, right, bottom, left)
                'personnel_id': pid,
                'confidence': conf,
                'encoding': enc,
            })
        return results
    except ImportError:
        return []
    except Exception as e:
        print(f'[Face] Detection error: {e}')
        return []


def add_encoding(personnel_id, face_encoding):
    """Add or update encoding in memory cache."""
    with _encodings_lock:
        _known_encodings[personnel_id] = face_encoding


def remove_encoding(personnel_id):
    """Remove encoding from memory cache."""
    with _encodings_lock:
        _known_encodings.pop(personnel_id, None)
