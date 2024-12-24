from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import requests
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pillow_avif
from PIL import Image, UnidentifiedImageError
import numpy as np
import onnxruntime as ort
import torchvision.transforms.functional as TVF
import cv2
import io
import tempfile
import json

app = FastAPI()

# Constants
MODEL_PATH = "joytag/model.onnx"
TOP_TAGS_PATH = "joytag/top_tags.txt"
SENSITIVITIIES_PATH = "sensitivities.json"
THRESHOLD = 0.4

# Global variables
session = None
top_tags = []
sensitivities = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_model():
    global session, top_tags, sensitivities

    try:
        # Load ONNX model
        session = ort.InferenceSession(
            MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Load top tags
        with open(TOP_TAGS_PATH, "r") as f:
            top_tags = [line.strip() for line in f.readlines() if line.strip()]

        with open(SENSITIVITIIES_PATH, "r") as f:
            sensitivities = json.load(f)

        print("ONNX model and top tags loaded successfully.")
    except Exception as e:
        print(f"Error loading model or top tags: {e}")
        raise RuntimeError("Failed to initialize model.")


# Helper function to prepare the image
def prepare_image(image: Image.Image, target_size: int) -> np.ndarray:
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    image_tensor = TVF.pil_to_tensor(padded_image).float() / 255.0
    image_tensor = TVF.normalize(
        image_tensor,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    return image_tensor.unsqueeze(0).numpy()


def sample_frames_from_webp(file: UploadFile, num_frames: int = 10) -> list:
    """
    Extracts frames from an animated WebP or treats it as a single image if not animated.
    """
    try:
        image = Image.open(file.file)
        frames = []

        # If animated, sample frames
        if getattr(image, "is_animated", False):
            frame_count = image.n_frames
            for i in np.linspace(0, frame_count - 1, num_frames, dtype=int):
                image.seek(i)
                frames.append(image.copy())
        else:
            # Single frame
            frames.append(image)

        return frames
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Failed to parse WebP file.")


def sample_frames(video_path: str, num_frames: int = 10) -> list:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_frames = []

    for i in np.linspace(0, frame_count - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(Image.fromarray(frame))

    cap.release()
    return sampled_frames


def process_files(content: bytes, extension: str):
    global session, top_tags, sensitivities

    try:
        if extension == "webp":
            # Handle WebP file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file.flush()
                file = UploadFile(filename=tmp_file.name, file=tmp_file)
                frames = sample_frames_from_webp(file)
        elif extension in {"mp4", "avi", "mov", "webm", "gif"}:
            # Handle video
            with tempfile.NamedTemporaryFile(delete=False) as tmp_video:
                tmp_video.write(content)
                video_path = tmp_video.name
            frames = sample_frames(video_path)
        else:
            # Handle standard image
            frames = [Image.open(io.BytesIO(content))]

        # Get ONNX model input shape
        input_shape = session.get_inputs()[0].shape
        target_size = input_shape[2]

        aggregated_scores = {}
        for frame in frames:
            image_array = prepare_image(frame, target_size)
            input_name = session.get_inputs()[0].name
            preds = session.run(None, {input_name: image_array})

            tag_preds = preds[0]  # Assuming predictions are in the first output
            scores = {top_tags[i]: float(tag_preds[0][i]) for i in range(len(top_tags))}
            for tag, score in scores.items():
                aggregated_scores[tag] = max(
                    aggregated_scores.get(tag, -float("inf")), score
                )

        predicted_tags = [
            tag
            for tag, score in aggregated_scores.items()
            if score > THRESHOLD
            or any(
                [
                    sensitive_tag in tag and score > sensitivity
                    for sensitive_tag, sensitivity in sensitivities.items()
                ]
            )
        ]

        sorted_tags = sorted(
            aggregated_scores.items(), key=lambda x: x[1], reverse=True
        )

        return {"tags": predicted_tags, "scores": sorted_tags}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    extension = file.filename.split(".")[-1].lower()
    return JSONResponse(content=process_files(content, extension))


@app.post("/upload-from-url/")
async def upload_from_url(url: str = Form(...)):
    try:
        # Download the file from the given URL
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
            "referer": url,
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail="Failed to download file from URL."
            )

        # Determine file type from the URL or response headers
        extension = url.split(".")[-1].lower()
        return JSONResponse(content=process_files(response.content, extension))
    except Exception as e:
        print(url)
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
