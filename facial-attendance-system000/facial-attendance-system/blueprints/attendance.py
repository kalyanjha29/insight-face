import os
import uuid
from datetime import date, datetime

import cv2
import numpy as np
from flask import (
    Blueprint, render_template, request, redirect, url_for, flash,
    current_app, session, abort
)

from extensions import db
from models import ClassSection, Student, AttendanceRecord
from blueprints.auth import login_required, current_teacher
from blueprints.classes import _owned_class_or_404
import face_engine

attendance_bp = Blueprint("attendance", __name__)


@attendance_bp.route("/classes/<int:class_id>/attendance/upload", methods=["GET", "POST"])
@login_required
def upload_photo(class_id):
    cls = _owned_class_or_404(class_id)
    students = Student.query.filter_by(class_id=cls.id).all()

    if not students:
        flash("Add students to this class before taking attendance.", "danger")
        return redirect(url_for("classes.class_detail", class_id=cls.id))
    if not cls.model_trained:
        flash("Train the recognition model for this class first.", "danger")
        return redirect(url_for("classes.class_detail", class_id=cls.id))

    if request.method == "POST":
        file = request.files.get("classroom_photo")
        session_date = request.form.get("session_date") or date.today().isoformat()

        if not file or not file.filename:
            flash("Please choose a classroom photo to upload.", "danger")
            return render_template("upload_photo.html", cls=cls)

        data = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            flash("Could not read that image file. Try a JPG or PNG.", "danger")
            return render_template("upload_photo.html", cls=cls)

        classroom_dir = current_app.config["CLASSROOM_PHOTOS_DIR"]
        os.makedirs(classroom_dir, exist_ok=True)
        raw_filename = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(classroom_dir, raw_filename), image)

        # --- Face Recognition Engine: detect + match against registered faces ---
        threshold = current_app.config["RECOGNITION_CONFIDENCE_THRESHOLD"]
        results = face_engine.recognize_classroom_photo(
            current_app.config["MODELS_DIR"], cls.id, image, threshold
        )

        id_to_name = {s.id: f"{s.name} ({s.roll_no})" for s in students}
        annotated = face_engine.draw_annotations(image, results, id_to_name)
        annotated_filename = f"{uuid.uuid4().hex}_annotated.jpg"
        cv2.imwrite(os.path.join(classroom_dir, annotated_filename), annotated)

        # Stash the pending (unconfirmed) recognition results for the review step
        session[f"pending_{cls.id}"] = {
            "session_date": session_date,
            "raw_filename": raw_filename,
            "annotated_filename": annotated_filename,
            "faces": [
                {"box": r["box"], "student_id": r["student_id"], "confidence": r["confidence"]}
                for r in results
            ],
        }
        return redirect(url_for("attendance.review", class_id=cls.id))

    return render_template("upload_photo.html", cls=cls, today=date.today().isoformat())


@attendance_bp.route("/classes/<int:class_id>/attendance/review")
@login_required
def review(class_id):
    cls = _owned_class_or_404(class_id)
    pending = session.get(f"pending_{cls.id}")
    if not pending:
        flash("No pending attendance photo. Please upload one first.", "warning")
        return redirect(url_for("attendance.upload_photo", class_id=cls.id))

    students = Student.query.filter_by(class_id=cls.id).order_by(Student.roll_no).all()
    recognized_ids = {f["student_id"] for f in pending["faces"] if f["student_id"] is not None}
    unrecognized_students = [s for s in students if s.id not in recognized_ids]

    return render_template(
        "recognition_result.html",
        cls=cls,
        pending=pending,
        students=students,
        unrecognized_students=unrecognized_students,
    )


@attendance_bp.route("/classes/<int:class_id>/attendance/confirm", methods=["POST"])
@login_required
def confirm(class_id):
    cls = _owned_class_or_404(class_id)
    pending = session.pop(f"pending_{cls.id}", None)
    if not pending:
        flash("Nothing to confirm — please upload a photo first.", "warning")
        return redirect(url_for("attendance.upload_photo", class_id=cls.id))

    session_date = datetime.strptime(pending["session_date"], "%Y-%m-%d").date()
    students = Student.query.filter_by(class_id=cls.id).all()

    # Build final present/absent decision per student from the (possibly corrected) form
    present_ids = set()
    confidences = {}
    for i, face in enumerate(pending["faces"]):
        chosen = request.form.get(f"face_{i}")
        if chosen and chosen != "unknown":
            sid = int(chosen)
            present_ids.add(sid)
            confidences[sid] = face.get("confidence")

    for key in request.form:
        if key.startswith("manual_present_"):
            present_ids.add(int(key.replace("manual_present_", "")))

    for student in students:
        status = "Present" if student.id in present_ids else "Absent"
        source = "auto" if student.id in confidences else ("manual" if status == "Present" else "auto")

        record = AttendanceRecord.query.filter_by(
            class_id=cls.id, student_id=student.id, session_date=session_date
        ).first()
        if record is None:
            record = AttendanceRecord(class_id=cls.id, student_id=student.id, session_date=session_date)
            db.session.add(record)

        record.status = status
        record.confidence = confidences.get(student.id)
        record.source = source
        record.classroom_photo = pending["annotated_filename"]
        record.marked_at = datetime.utcnow()

    db.session.commit()
    flash(f"Attendance saved for {session_date.isoformat()}.", "success")
    return redirect(url_for("attendance.report", class_id=cls.id))


@attendance_bp.route("/classes/<int:class_id>/attendance/report")
@login_required
def report(class_id):
    cls = _owned_class_or_404(class_id)
    records = (
        AttendanceRecord.query.filter_by(class_id=cls.id)
        .order_by(AttendanceRecord.session_date.desc(), AttendanceRecord.student_id)
        .all()
    )

    by_date = {}
    for r in records:
        by_date.setdefault(r.session_date, []).append(r)

    return render_template("attendance_report.html", cls=cls, by_date=sorted(by_date.items(), reverse=True))


@attendance_bp.route("/attendance/<int:record_id>/toggle", methods=["POST"])
@login_required
def toggle(record_id):
    record = AttendanceRecord.query.get_or_404(record_id)
    cls = _owned_class_or_404(record.class_id)
    record.status = "Absent" if record.status == "Present" else "Present"
    record.source = "manual"
    db.session.commit()
    flash(f"Marked {record.student.name} as {record.status} for {record.session_date}.", "info")
    return redirect(url_for("attendance.report", class_id=cls.id))
