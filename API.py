import os
import cv2
import numpy as np
import json
import tempfile
import gc
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic, HAND_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec

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
holistic_model = Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.3
)

def draw_hand_landmarks(frame, results):
    if results.left_hand_landmarks and not results.right_hand_landmarks:
        draw_landmarks(frame, results.left_hand_landmarks, HAND_CONNECTIONS,
                       DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                       DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    elif results.right_hand_landmarks and not results.left_hand_landmarks:
        draw_landmarks(frame, results.right_hand_landmarks, HAND_CONNECTIONS,
                       DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                       DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    elif results.left_hand_landmarks and results.right_hand_landmarks:
        draw_landmarks(frame, results.left_hand_landmarks, HAND_CONNECTIONS,
                       DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                       DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        draw_landmarks(frame, results.right_hand_landmarks, HAND_CONNECTIONS,
                       DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                       DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

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

def rotate_video(input_path, output_path, camera: int = 2):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if camera == 2:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif camera == 1:
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = frame

        out.write(rotated)

    cap.release()
    out.release()

def interpolate_keypoints(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length == target_length:
        return keypoints
    indices = np.linspace(0, current_length - 1, target_length)
    interpolated_keypoints = []
    for i in indices:
        lower_idx = int(np.floor(i))
        upper_idx = int(np.ceil(i))
        weight = i - lower_idx
        if lower_idx == upper_idx:
            interpolated_keypoints.append(keypoints[lower_idx])
        else:
            interpolated_point = (1 - weight) * np.array(keypoints[lower_idx]) + weight * np.array(keypoints[upper_idx])
            interpolated_keypoints.append(interpolated_point.tolist())
    return interpolated_keypoints

def normalize_sequence(keypoints, target_length=15):
    current_length = len(keypoints)
    if current_length < target_length:
        return np.array(interpolate_keypoints(keypoints, target_length))
    elif current_length > target_length:
        step = current_length / target_length
        indices = np.arange(0, current_length, step).astype(int)[:target_length]
        return np.array([keypoints[i] for i in indices])
    else:
        return np.array(keypoints)

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

        frame = cv2.resize(frame, (640, 480))
        results = mediapipe_detection(frame)

        # --- Captura de keypoints ---
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

    return [pred_word] if pred_word else []

@app.post("/predict")
async def predict(video: UploadFile = File(...), camera: int = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        tmp.write(await video.read())

    rotated_path = video_path.replace(".mp4", "_rotated.mp4")
    rotate_video(video_path, rotated_path, camera)

    try:
        result = evaluate_video(rotated_path, model, word_ids)
    finally:
        os.remove(video_path)
        if os.path.exists(rotated_path):
            os.remove(rotated_path)

    return JSONResponse(content={"prediction": result})

@app.post("/predictASL")
async def predict_asl(video: UploadFile = File(...), camera: int = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        tmp.write(await video.read())

    rotated_path = video_path.replace(".mp4", "_rotated.mp4")
    rotate_video(video_path, rotated_path, camera)

    try:
        result = evaluate_video(rotated_path, model_asl, word_ids_asl)
    finally:
        os.remove(video_path)
        if os.path.exists(rotated_path):
            os.remove(rotated_path)

    return JSONResponse(content={"prediction": result})

