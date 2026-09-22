from functools import wraps

from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app

from extensions import db
from models import Teacher

auth_bp = Blueprint("auth", __name__)


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if "teacher_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("auth.login"))
        return view_func(*args, **kwargs)

    return wrapped


def current_teacher():
    teacher_id = session.get("teacher_id")
    if not teacher_id:
        return None
    return Teacher.query.get(teacher_id)


@auth_bp.route("/")
def index():
    if "teacher_id" in session:
        return redirect(url_for("classes.dashboard"))
    return redirect(url_for("auth.login"))


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        teacher = Teacher.query.filter_by(username=username).first()
        # Auth API: verify credentials -> generate session token
        if teacher and teacher.check_password(password):
            session["teacher_id"] = teacher.id
            session["teacher_name"] = teacher.name
            flash(f"Welcome back, {teacher.name}!", "success")
            return redirect(url_for("classes.dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """First-run teacher account creation (stands in for an admin-provisioned account)."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not (name and username and password):
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if Teacher.query.filter_by(username=username).first():
            flash("That username is already taken.", "danger")
            return render_template("register.html")

        teacher = Teacher(name=name, username=username)
        teacher.set_password(password)
        db.session.add(teacher)
        db.session.commit()
        flash("Account created — please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
