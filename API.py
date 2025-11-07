import os
import cv2
import numpy as np
import json
import tempfile
import gc

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from keras.models import load_model
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic

app = FastAPI()

MODEL_FRAMES = 15
ROOT_PATH = os.getcwd()
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")

# --- Modelos y JSONs ---
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"modelLSP_v2_{MODEL_FRAMES}.keras")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words_asl.json")

MODEL_PATH_ASL = os.path.join(MODEL_FOLDER_PATH, f"modelASL_v2_{MODEL_FRAMES}.keras")
WORDS_JSON_PATH_ASL = os.path.join(MODEL_FOLDER_PATH, "words.json")

# --- Cargar modelos y word_ids ---
model = load_model(MODEL_PATH)
with open(WORDS_JSON_PATH, 'r') as f:
    word_ids = json.load(f).get('word_ids')

model_asl = load_model(MODEL_PATH_ASL)
with open(WORDS_JSON_PATH_ASL, 'r') as f:
    word_ids_asl = json.load(f).get('word_ids')

# --- Holistic global ---
holistic_model = Holistic(static_image_mode=False,
                          model_complexity=0,
                          smooth_landmarks=True,
                          enable_segmentation=False,
                          refine_face_landmarks=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

# --- Funciones auxiliares ---
def mediapipe_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    return holistic_model.process(image)

def there_hand(results):
    return results.left_hand_landmarks or results.right_hand_landmarks

def extract_hand_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def normalize_sequence(sequence, target_length=15):
    current_length = len(sequence)
    if current_length == target_length:
        return np.array(sequence)
    indices = np.linspace(0, current_length-1, target_length)
    normalized = []
    for i in indices:
        lower = int(np.floor(i))
        upper = int(np.ceil(i))
        w = i - lower
        if lower == upper:
            normalized.append(sequence[lower])
        else:
            normalized.append((1-w)*np.array(sequence[lower]) + w*np.array(sequence[upper]))
    return np.array(normalized)

# --- Procesamiento del video (generalizado) ---
def evaluate_video(video_path, model, word_ids, threshold=0.8, min_frames=5, delay_frames=3):
    kp_seq = []
    pred_word = None
    count_frame = 0
    fix_frames = 0
    recording = False
    frame_skip = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (224, 224))
        results = mediapipe_detection(frame)

        if there_hand(results) or recording:
            recording = False
            count_frame += 1
            if count_frame > 1:
                kp_seq.append(extract_hand_keypoints(results))
        else:
            if count_frame >= min_frames:
                fix_frames += 1
                if fix_frames < delay_frames:
                    recording = True
                    continue

                kp_seq = kp_seq[:-(delay_frames)]
                if len(kp_seq) > 0:
                    kp_norm = normalize_sequence(kp_seq, MODEL_FRAMES).astype(np.float16)
                    res = model.predict(np.expand_dims(kp_norm, axis=0), verbose=0)[0]
                    idx = int(np.argmax(res))
                    conf = float(res[idx])
                    if conf > threshold:
                        pred_word = word_ids[idx]

                kp_seq = []
                count_frame = 0
                fix_frames = 0
                recording = False

        del results

    cap.release()
    gc.collect()
    tf.keras.backend.clear_session()

    return [pred_word] if pred_word else []

# --- API ---
@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        contents = await video.read()
        tmp.write(contents)

    try:
        result = evaluate_video(video_path, model, word_ids)
    finally:
        os.remove(video_path)

    return JSONResponse(content={"prediction": result[0] if result else None})

@app.post("/predictASL")
async def predict_asl(video: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        contents = await video.read()
        tmp.write(contents)

    try:
        result = evaluate_video(video_path, model_asl, word_ids_asl)
    finally:
        os.remove(video_path)

    return JSONResponse(content={"prediction": result[0] if result else None})

