import os
import uuid

import cv2
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, abort

from extensions import db
from models import ClassSection, Student, StudentPhoto
from blueprints.auth import login_required, current_teacher
import face_engine

classes_bp = Blueprint("classes", __name__)


def _owned_class_or_404(class_id):
    cls = ClassSection.query.get_or_404(class_id)
    if cls.teacher_id != current_teacher().id:
        abort(403)
    return cls


@classes_bp.route("/dashboard")
@login_required
def dashboard():
    teacher = current_teacher()
    classes = ClassSection.query.filter_by(teacher_id=teacher.id).order_by(ClassSection.created_at.desc()).all()
    return render_template("dashboard.html", classes=classes, teacher=teacher)


@classes_bp.route("/classes/new", methods=["GET", "POST"])
@login_required
def new_class():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        section = request.form.get("section", "").strip()
        if not name:
            flash("Class name is required.", "danger")
            return render_template("new_class.html")
        cls = ClassSection(teacher_id=current_teacher().id, name=name, section=section)
        db.session.add(cls)
        db.session.commit()
        flash(f"Class '{cls.display_name}' created.", "success")
        return redirect(url_for("classes.class_detail", class_id=cls.id))
    return render_template("new_class.html")


@classes_bp.route("/classes/<int:class_id>")
@login_required
def class_detail(class_id):
    cls = _owned_class_or_404(class_id)
    students = Student.query.filter_by(class_id=cls.id).order_by(Student.roll_no).all()
    return render_template("class_detail.html", cls=cls, students=students)


@classes_bp.route("/classes/<int:class_id>/delete", methods=["POST"])
@login_required
def delete_class(class_id):
    cls = _owned_class_or_404(class_id)
    db.session.delete(cls)
    db.session.commit()
    flash("Class deleted.", "info")
    return redirect(url_for("classes.dashboard"))


@classes_bp.route("/classes/<int:class_id>/students/new", methods=["GET", "POST"])
@login_required
def new_student(class_id):
    cls = _owned_class_or_404(class_id)

    if request.method == "POST":
        roll_no = request.form.get("roll_no", "").strip()
        name = request.form.get("name", "").strip()
        files = [f for f in request.files.getlist("photos") if f and f.filename]

        if not (roll_no and name):
            flash("Roll number and name are required.", "danger")
            return render_template("new_student.html", cls=cls)
        if not files:
            flash("Please upload at least one reference photo of the student's face.", "danger")
            return render_template("new_student.html", cls=cls)

        student = Student(class_id=cls.id, roll_no=roll_no, name=name)
        db.session.add(student)
        db.session.flush()  # get student.id before commit

        photos_dir = current_app.config["STUDENT_PHOTOS_DIR"]
        os.makedirs(photos_dir, exist_ok=True)

        saved, skipped = 0, 0
        for f in files[:5]:  # cap at 5 reference photos per student
            data = np.frombuffer(f.read(), np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                skipped += 1
                continue
            if not face_engine.detect_faces(image):
                skipped += 1
                continue
            filename = f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(os.path.join(photos_dir, filename), image)
            db.session.add(StudentPhoto(student_id=student.id, filename=filename))
            saved += 1

        if saved == 0:
            db.session.rollback()
            flash("No usable face was detected in the uploaded photo(s). Please try clearer, front-facing photos.", "danger")
            return render_template("new_student.html", cls=cls)

        db.session.commit()
        msg = f"Student '{name}' added with {saved} reference photo(s)."
        if skipped:
            msg += f" ({skipped} photo(s) skipped — no face detected.)"
        flash(msg, "success")
        return redirect(url_for("classes.class_detail", class_id=cls.id))

    return render_template("new_student.html", cls=cls)


@classes_bp.route("/students/<int:student_id>/delete", methods=["POST"])
@login_required
def delete_student(student_id):
    student = Student.query.get_or_404(student_id)
    cls = _owned_class_or_404(student.class_id)
    db.session.delete(student)
    db.session.commit()
    flash("Student removed.", "info")
    return redirect(url_for("classes.class_detail", class_id=cls.id))


@classes_bp.route("/classes/<int:class_id>/train", methods=["POST"])
@login_required
def train_class(class_id):
    """Trains the Face Recognition Engine on all registered students' photos —
    extracts a face embedding from each reference photo and saves them for
    this class's recognizer to match against later."""
    cls = _owned_class_or_404(class_id)
    students = Student.query.filter_by(class_id=cls.id).all()

    photos_dir = current_app.config["STUDENT_PHOTOS_DIR"]
    student_encodings = {}
    for student in students:
        encodings = []
        for photo in student.photos:
            path = os.path.join(photos_dir, photo.filename)
            image = cv2.imread(path)
            if image is None:
                continue
            encoding = face_engine.extract_largest_face_encoding(image)
            if encoding is not None:
                encodings.append(encoding)
        if encodings:
            student_encodings[student.id] = encodings

    if not student_encodings:
        flash("Add students with clear face photos before training.", "danger")
        return redirect(url_for("classes.class_detail", class_id=cls.id))

    try:
        num_faces, num_students = face_engine.train_class_recognizer(
            current_app.config["MODELS_DIR"], cls.id, student_encodings
        )
    except ValueError as e:
        flash(str(e), "danger")
        return redirect(url_for("classes.class_detail", class_id=cls.id))

    flash(f"Recognition model trained on {num_faces} photo(s) across {num_students} student(s).", "success")
    return redirect(url_for("classes.class_detail", class_id=cls.id))
