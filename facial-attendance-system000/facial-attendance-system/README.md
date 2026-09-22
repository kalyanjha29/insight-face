# College Classroom Facial Recognition Attendance System

A working implementation of the architecture diagram: Teacher Login →
Class & Student Manager → Face Recognition Engine → Attendance Manager.

## Stack actually used (and why)

| Layer | Diagram said | Built with | Why |
|---|---|---|---|
| Frontend | React | Server-rendered HTML (Jinja2) + Bootstrap 5 | Zero build step — just run one Python process, no `npm install` needed to demo it |
| Backend | Python FastAPI/Flask | **Flask**, organized as blueprints matching the 4 diagram modules | Matches the diagram's suggested stack directly |
| Database | PostgreSQL/MySQL | **SQLite** via SQLAlchemy | Zero-config; the SQLAlchemy models map 1:1 to the diagram's "Teacher DB" / "Student & Facial Data DB" / "Attendance DB". Swapping in Postgres later is a one-line `SQLALCHEMY_DATABASE_URI` change. |
| Face Recognition | OpenCV + face-recognition / InsightFace | **MTCNN (detection) + FaceNet (recognition)**, via `mtcnn` + `keras-facenet` on `tensorflow-cpu` | Real CNN-based pipeline. MTCNN is a proper 3-stage cascaded CNN face detector; FaceNet is a deep CNN trained with triplet loss that maps faces into a 512-d embedding space where distance directly reflects facial similarity. A dlib/`face_recognition`-based version was also built and works well, but `dlib`'s only route to a fast install is a third-party prebuilt wheel (`dlib-bin`); MTCNN+FaceNet needed nothing but mainline, official PyPI packages. Both are documented below. |

An earlier version of this project used OpenCV's LBPH (Local Binary Pattern
Histogram) recognizer, which compares raw pixel-texture statistics rather
than learned face features — it's fast and dependency-light, but noticeably
weaker at telling similar-looking people apart or handling lighting/angle
variation. MTCNN+FaceNet is a real deep-learning pipeline and recognizes
faces far more reliably.

## Project layout

```
app.py                  entry point — creates the Flask app, registers blueprints
config.py               paths, DB URI, recognition distance threshold
extensions.py           the shared SQLAlchemy() instance
models.py               Teacher, ClassSection, Student, StudentPhoto, AttendanceRecord
face_engine.py          MTCNN detection + FaceNet embeddings/matching (Module 3: Face Recognition Engine)
blueprints/
  auth.py               Module 1: Authentication (login / register / logout)
  classes.py             Module 2: Class & Student Manager (create classes, register students, train model)
  attendance.py          Modules 3+4: upload classroom photo -> recognize -> review/correct -> save -> report
templates/               Jinja2 + Bootstrap 5 pages
static/
  uploads/students/      registered reference face photos
  uploads/classroom/     uploaded classroom photos + annotated review images
  models/                one embeddings .json per class (FaceNet vectors keyed by student_id)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000, click **Register as teacher**, then log in.

**Note on install size:** `tensorflow-cpu` is a ~1GB install (the price of a
real CNN pipeline with no GPU needed). The `requirements.txt` pins
`numpy==1.26.4` and `scipy<1.13` deliberately — newer versions of both
default to numpy 2.x, which breaks this TensorFlow version's compiled
extensions. Don't `pip install --upgrade` these individually.

## Using it

1. **Register / Login** — create a teacher account, log in.
2. **New Class** — create a class/section.
3. **Add Student** — for each student, enter roll no. + name and upload 2–3
   clear, front-facing reference photos. Each photo must contain exactly one
   detectable face (photos with no detectable face are skipped automatically).
4. **Train Recognition Model** — click this once you've added students (or
   again any time you add more). This runs FaceNet on every registered
   student photo and saves the resulting embeddings for this class.
5. **Take Attendance** — upload a classroom photo. The Face Recognition
   Engine detects every face with MTCNN, matches each one against the
   trained embeddings, and shows you an annotated review image (green box =
   recognized, red = unknown).
6. **Review & correct** — each detected face has a dropdown so you can fix a
   wrong/uncertain match, and any student the camera missed can be checked
   off manually. Click **Confirm & Save Attendance**.
7. **Attendance Report** — per-class, per-date table of Present/Absent with
   a one-click **Mark Present/Absent** toggle for later corrections.

## Notes on the recognition approach

- Detection: MTCNN (P-Net → R-Net → O-Net cascade). Handles a reasonable
  range of angles/scales better than a Haar cascade, and returns confidence
  scores per detection.
- Recognition: FaceNet (via `keras-facenet`) embeds each detected face into
  a 512-d vector. A detected face is matched to whichever registered
  student's reference embeddings are closest (Euclidean distance) — lower
  distance is a better match. Matches worse than
  `RECOGNITION_CONFIDENCE_THRESHOLD` (see `config.py`, default `0.9`) are
  reported as "Unknown" rather than forced into a guess. Tune this down
  (e.g. `0.75`) if you're getting false matches, or up if too many known
  faces come back "Unknown".
- Recognition quality still depends on reference photo quality (frontal,
  well-lit, one face per photo) and classroom photo quality (faces not too
  small/angled/backlit). For even higher accuracy on harder photos, the
  same `detect_faces()` / `recognize_classroom_photo()` interface in
  `face_engine.py` could be swapped for ArcFace or InsightFace without
  touching any other file — the rest of the app only depends on those two
  functions' input/output shapes.

## Extending toward the original diagram

- Swap SQLite → PostgreSQL/MySQL: change `SQLALCHEMY_DATABASE_URI` in
  `config.py`; models are already ORM-based so nothing else changes.
- Swap templates → React SPA: the blueprint logic in `classes.py` /
  `attendance.py` can be exposed as pure JSON endpoints (`jsonify(...)`
  instead of `render_template(...)`) with minimal changes, since the
  business logic is already separated from presentation.

