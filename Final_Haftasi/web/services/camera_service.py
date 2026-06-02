"""Camera streaming service — MJPEG generator for live feed."""
import cv2
import time
import threading
import numpy as np

_active_captures = {}
_capture_lock = threading.Lock()


def _get_capture(stream_url):
    with _capture_lock:
        key = str(stream_url)
        if key in _active_captures:
            cap = _active_captures[key]
            if cap.isOpened():
                return cap
        # Try int index first
        src = int(stream_url) if stream_url.isdigit() else stream_url
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            _active_captures[key] = cap
        return cap


def _placeholder_frame(message='Kamera Bağlanamıyor'):
    """Return a dark placeholder frame with message."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (18, 25, 38)
    cv2.putText(frame, message, (60, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
    cv2.putText(frame, 'ARCHISAFE', (220, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 215), 1)
    return frame


class CameraService:
    @staticmethod
    def generate_frames(camera, enable_yolo=False):
        """MJPEG generator for Flask Response."""
        from .yolo_service import detect_frame, annotate_frame

        stream_url = camera.stream_url or '0'
        cap = _get_capture(stream_url)

        consecutive_fails = 0
        while True:
            if not cap.isOpened():
                frame = _placeholder_frame(f'{camera.name} — Bağlantı Yok')
            else:
                ret, frame = cap.read()
                if not ret:
                    consecutive_fails += 1
                    if consecutive_fails > 30:
                        frame = _placeholder_frame(f'{camera.name} — Sinyal Yok')
                    else:
                        time.sleep(0.05)
                        continue
                else:
                    consecutive_fails = 0

                    if enable_yolo:
                        try:
                            detections = detect_frame(frame)
                            frame = annotate_frame(frame, detections)
                        except Exception:
                            pass

            # Encode JPEG
            ret2, buffer = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if not ret2:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                buffer.tobytes() +
                b'\r\n'
            )
            time.sleep(0.04)  # ~25 FPS

    @staticmethod
    def capture_snapshot(camera):
        """Capture a single frame and return as JPEG bytes."""
        stream_url = camera.stream_url or '0'
        cap = _get_capture(stream_url)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                return buf.tobytes()
        return None

    @staticmethod
    def release_all():
        with _capture_lock:
            for cap in _active_captures.values():
                cap.release()
            _active_captures.clear()
