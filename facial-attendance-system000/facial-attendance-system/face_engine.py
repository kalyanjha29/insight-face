"""
Face Recognition Engine (InsightFace)
-------------------------------------
Implements module 3 of the architecture: detects faces in an image and
matches them against a class's registered students using InsightFace
(RetinaFace for detection + ArcFace for 512-d normalized embeddings).
"""
import json
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Recommended Cosine Similarity threshold for ArcFace.
# Range is 0.0 to 1.0 (higher = stricter match).
DEFAULT_SIMILARITY_THRESHOLD = 0.35

_app = None


def _get_app():
    """Lazily instantiate InsightFace FaceAnalysis once."""
    global _app
    if _app is None:
        # buffalo_l includes RetinaFace for detection and ArcFace for embeddings
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # Increased det_size to (1280, 1280) to detect small faces in group photos
        _app.prepare(ctx_id=0, det_size=(1280, 1280))
    return _app


def detect_faces(image_bgr):
    """Detect faces in a BGR image using InsightFace. Returns a list of (x, y, w, h) boxes."""
    app = _get_app()
    faces = app.get(image_bgr)
    boxes = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        x, y = max(0, int(x1)), max(0, int(y1))
        w, h = max(0, int(x2 - x1)), max(0, int(y2 - y1))
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def extract_largest_face_encoding(image_bgr):
    """
    Used during student registration. Finds the largest detected face 
    and returns its 512-d normalized embedding array, or None if no face is found.
    """
    app = _get_app()
    faces = app.get(image_bgr)
    if not faces:
        return None

    # Get face with largest bounding box area
    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    
    # Return normalized 512-d embedding
    embedding = largest_face.embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def _model_path(models_dir, class_id):
    return os.path.join(models_dir, f"class_{class_id}_embeddings.json")


def train_class_recognizer(models_dir, class_id, student_encodings):
    """Saves student reference embeddings for the class to a single JSON file."""
    os.makedirs(models_dir, exist_ok=True)

    data = {}
    total = 0
    for student_id, encs in student_encodings.items():
        if not encs:
            continue
        data[str(student_id)] = [e.tolist() for e in encs]
        total += len(encs)

    if not data:
        raise ValueError("No registered face photos available to train on for this class.")

    with open(_model_path(models_dir, class_id), "w") as f:
        json.dump(data, f)

    return total, len(data)


def load_class_recognizer(models_dir, class_id):
    """Returns {student_id (int): [np.ndarray(512,), ...]} or None if untrained."""
    path = _model_path(models_dir, class_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): [np.array(e) for e in v] for k, v in raw.items()}


def compute_cosine_similarity(emb1, emb2):
    """Computes cosine similarity between two 512-d embeddings."""
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def recognize_classroom_photo(models_dir, class_id, image_bgr, threshold=DEFAULT_SIMILARITY_THRESHOLD):
    """
    Detects faces in a classroom photo using InsightFace and matches each against 
    registered student embeddings using Cosine Similarity.
    """
    student_db = load_class_recognizer(models_dir, class_id)
    app = _get_app()
    faces = app.get(image_bgr)

    print(f"\n--- InsightFace Processing Photo: Detected {len(faces)} face(s) ---")

    results = []
    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        box = (max(0, int(x1)), max(0, int(y1)), max(0, int(x2 - x1)), max(0, int(y2 - y1)))
        
        # Extract and normalize face embedding
        enc = face.embedding
        norm = np.linalg.norm(enc)
        if norm > 0:
            enc = enc / norm

        if enc is None or not student_db:
            results.append({"box": box, "student_id": None, "confidence": None})
            continue

        best_student, max_similarity = None, -1.0
        for student_id, ref_encodings in student_db.items():
            sims = [compute_cosine_similarity(enc, ref) for ref in ref_encodings]
            sim = max(sims)
            
            # Print similarity scores to terminal for debugging
            print(f"Face #{idx+1} vs Student ID {student_id} -> Max Similarity: {sim:.3f}")
            
            if sim > max_similarity:
                max_similarity = sim
                best_student = student_id

        matched_id = int(best_student) if best_student is not None else None

        if max_similarity >= threshold:
            print(f"  ==> MATCHED: Face #{idx+1} -> Student ID {matched_id} (Score: {max_similarity:.3f})")
            results.append({"box": box, "student_id": matched_id, "confidence": round(float(max_similarity), 3)})
        else:
            print(f"  ==> UNKNOWN: Face #{idx+1} (Best Score: {max_similarity:.3f} < Threshold {threshold})")
            results.append({
                "box": box,
                "student_id": None,
                "confidence": round(float(max_similarity), 3) if max_similarity >= 0 else None,
            })

    return results


def draw_annotations(image_bgr, results, id_to_name):
    """Draws bounding boxes + labels on a copy of the image for review."""
    annotated = image_bgr.copy()
    for r in results:
        x, y, w, h = r["box"]
        student_id = r["student_id"]
        conf = r["confidence"]
        
        if student_id is not None:
            color = (46, 139, 87)  # green (BGR)
            name = id_to_name.get(student_id, f"ID {student_id}")
            label = f"{name} ({conf})" if conf else name
        else:
            color = (0, 0, 220)  # red
            label = "Unknown"

        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        text_y = y - 10 if y - 10 > 10 else y + h + 20
        cv2.putText(annotated, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated