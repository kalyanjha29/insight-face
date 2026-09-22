from datetime import datetime, date

from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db


class Teacher(db.Model):
    """Teacher Database — teacher details + password hash."""
    __tablename__ = "teachers"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    classes = db.relationship("ClassSection", backref="teacher", cascade="all, delete-orphan")

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password_hash, raw_password)


class ClassSection(db.Model):
    """A class / section a teacher manages (Class & Student Manager)."""
    __tablename__ = "class_sections"

    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey("teachers.id"), nullable=False)
    name = db.Column(db.String(120), nullable=False)          # e.g. "B.Sc CS Semester IV"
    section = db.Column(db.String(50), nullable=True)         # e.g. "Section A"
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    students = db.relationship("Student", backref="class_section", cascade="all, delete-orphan")
    attendance_records = db.relationship("AttendanceRecord", backref="class_section", cascade="all, delete-orphan")

    @property
    def display_name(self):
        return f"{self.name} ({self.section})" if self.section else self.name

    @property
    def model_trained(self):
        import os
        from flask import current_app
        model_path = os.path.join(current_app.config["MODELS_DIR"], f"class_{self.id}_embeddings.json")
        return os.path.exists(model_path)


class Student(db.Model):
    """Student & Facial Data Database — student info + facial embeddings (photo-backed)."""
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey("class_sections.id"), nullable=False)
    roll_no = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    photos = db.relationship("StudentPhoto", backref="student", cascade="all, delete-orphan")
    attendance_records = db.relationship("AttendanceRecord", backref="student", cascade="all, delete-orphan")

    @property
    def photo_count(self):
        return len(self.photos)


class StudentPhoto(db.Model):
    """One registered facial-reference photo for a student. The face crop stored
    here is what the LBPH recognizer is trained on (this stands in for a
    'facial embedding' in the architecture diagram)."""
    __tablename__ = "student_photos"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AttendanceRecord(db.Model):
    """Attendance Database — attendance records with date/time and class/section."""
    __tablename__ = "attendance_records"

    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey("class_sections.id"), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    session_date = db.Column(db.Date, default=date.today, nullable=False)
    marked_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default="Absent")  # Present / Absent
    confidence = db.Column(db.Float, nullable=True)       # recognition confidence (lower = better, LBPH distance)
    source = db.Column(db.String(20), default="auto")     # auto (face recognition) / manual (teacher correction)
    classroom_photo = db.Column(db.String(255), nullable=True)

    __table_args__ = (
        db.UniqueConstraint("class_id", "student_id", "session_date", name="uq_attendance_per_day"),
    )
