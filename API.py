import os
import cv2
import base64
import numpy as np
import json
import gc
import uvicorn
from typing import List
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic, HAND_CONNECTIONS

app = FastAPI()

# ---------- SETTINGS ----------
MODEL_FRAMES = 15
THRESHOLD = 0.7

ROOT_PATH = os.getcwd()
MODEL_FOLDER_PATH = os.path.join(ROOT_PATH, "models")

# ---------- Modelos y JSONs ----------
MODEL_PATH = os.path.join(MODEL_FOLDER_PATH, f"modelLSP_v2_{MODEL_FRAMES}.keras")
WORDS_JSON_PATH = os.path.join(MODEL_FOLDER_PATH, "words_asl.json")

MODEL_PATH_ASL = os.path.join(MODEL_FOLDER_PATH, f"modelASL_v2_{MODEL_FRAMES}.keras")
WORDS_JSON_PATH_ASL = os.path.join(MODEL_FOLDER_PATH, "words.json")

# ---------- Cargar modelos ----------
model = load_model(MODEL_PATH)
with open(WORDS_JSON_PATH, "r") as f:
    word_ids = json.load(f).get("word_ids")

model_asl = load_model(MODEL_PATH_ASL)
with open(WORDS_JSON_PATH_ASL, "r") as f:
    word_ids_asl = json.load(f).get("word_ids")

# ---------- Inicializar MediaPipe ----------
holistic = Holistic(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=False,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Helpers ----------
def strip_base64_prefix(s: str) -> str:
    if "," in s:
        return s.split(",", 1)[1]
    return s

def image_from_base64(b64str: str):
    try:
        b64 = strip_base64_prefix(b64str)
        data = base64.b64decode(b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def mediapipe_detection(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)
    return results

def draw_hand_landmarks(image, results):
    h, w, _ = image.shape

    if results.left_hand_landmarks:
        for connection in HAND_CONNECTIONS:
            start = results.left_hand_landmarks.landmark[connection[0]]
            end = results.left_hand_landmarks.landmark[connection[1]]
            cv2.line(image,
                     (int(start.x * w), int(start.y * h)),
                     (int(end.x * w), int(end.y * h)),
                     (0, 255, 0), 2)
        for lm in results.left_hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)

    if results.right_hand_landmarks:
        for connection in HAND_CONNECTIONS:
            start = results.right_hand_landmarks.landmark[connection[0]]
            end = results.right_hand_landmarks.landmark[connection[1]]
            cv2.line(image,
                     (int(start.x * w), int(start.y * h)),
                     (int(end.x * w), int(end.y * h)),
                     (0, 0, 255), 2)
        for lm in results.right_hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

def extract_keypoints_from_results(results):
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# ---------- Core predict ----------
def predict_sequence(frames_b64: List[str], model, word_ids, camera_id=1):
    kp_seq = []
    for i, b64 in enumerate(frames_b64):
        img = image_from_base64(b64)
        if img is None:
            print(f"[Frame {i}] No se pudo decodificar.")
            continue

        # Si es frontal (2) → gira 90° a la izquierda
        # Si es trasera (1) → gira 270° a la izquierda
        if camera_id == 2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # o directamente:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        results = mediapipe_detection(img)
        kp = extract_keypoints_from_results(results)
        kp_seq.append(kp)

        # Mostrar frames
        #draw_hand_landmarks(img, results)
        #cv2.putText(img, f"Frame {i+1}", (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.imshow("Detección de Keypoints", img)
        #cv2.waitKey(80)

    #cv2.destroyAllWindows()

    if not kp_seq:
        return {"error": "No valid frames with hands detected"}

    kp_seq = np.array(kp_seq, dtype=np.float16)

    pred = model.predict(np.expand_dims(kp_seq, axis=0), verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    word = word_ids[idx] if conf > THRESHOLD else None

    return {"prediction": [word], "confidence": conf}


# ---------- Endpoints ----------
@app.post("/predict_sequence")
async def predict_sequence_lsp(payload: dict = Body(...)):
    try:
        frames_b64 = payload.get("frames") or payload.get("sequence")
        camera_id = payload.get("camera", 1)  # 1 = trasera, 2 = frontal NUEVO

        if not frames_b64:
            return JSONResponse({"error": "No frames provided"}, status_code=400)

        result = predict_sequence(frames_b64, model, word_ids, camera_id)
        return JSONResponse(result)
    finally:
        gc.collect()

@app.post("/predict_sequence_asl")
async def predict_sequence_asl(payload: dict = Body(...)):
    try:
        frames_b64 = payload.get("frames") or payload.get("sequence")
        camera_id = payload.get("camera", 1)  # 1 = trasera, 2 = frontal NUEVO


        if not frames_b64:
            return JSONResponse({"error": "No frames provided"}, status_code=400)

        result = predict_sequence(frames_b64, model_asl, word_ids_asl, camera_id)
        return JSONResponse(result)
    finally:
        gc.collect()

