# Coconut Guardian

Detect coconut tree diseases from an uploaded image using a TensorFlow/Keras model, with an optional Grad-CAM heatmap visualization.

![Coconut Guardian banner](assets/banner.png)

## Demo

- **Live app (Cloud Run)**: `https://coconut-guardian-1079917738080.asia-south1.run.app`
- **Demo video**: add your link here (YouTube/Drive) or attach `assets/demo.mp4` and link it below.

> If you don’t have media yet, see `assets/README.md` for what to export and where to place it.

## Features

- Upload an image and get a predicted disease label + confidence.
- Grad-CAM heatmap overlay (when supported by the model layer configuration).
- Simple Flask UI.

## Tech stack

- **Backend**: Flask
- **ML**: TensorFlow / Keras
- **Image**: Pillow, OpenCV (headless)
- **Deployment**: Docker + Google Cloud Run

## Project structure

```text
.
├─ app.py
├─ Dockerfile
├─ requirements.txt
├─ final_model.h5
├─ templates/
├─ static/
│  ├─ style.css
│  ├─ uploads/
│  └─ heatmaps/
└─ assets/
   ├─ banner.png
   ├─ screenshot-home.png
   ├─ screenshot-result.png
   └─ demo.mp4
```

## Run locally

### 1) Create a virtual environment (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Start the app

```powershell
$env:PORT=8080
python .\app.py
```

Open `http://localhost:8080`.

## Run with Docker

```powershell
docker build -t coconut-guardian .
docker run --rm -p 8080:8080 coconut-guardian
```

Open `http://localhost:8080`.

## Deploy to Google Cloud Run

This repo is compatible with Cloud Run’s container contract (listens on `PORT`, binds to `0.0.0.0`).

Typical deploy flow:

```bash
# Example (you can also deploy from the Console)
gcloud run deploy coconut-guardian \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated
```

Recommended Cloud Run settings for TensorFlow apps:

- **Memory**: 2GiB+ (model loading/inference can OOM on small sizes)
- **Timeout**: 300s (uploads + inference + Grad-CAM can be slow)
- **Min instances**: 1 (avoid cold-start “not working” moments)

## Screenshots

![Home](assets/screenshot-home.png)
![Result](assets/screenshot-result.png)

## Notes

- The model file `final_model.h5` must be present at runtime.
- Uploaded images and generated heatmaps are saved under `static/uploads` and `static/heatmaps`.

