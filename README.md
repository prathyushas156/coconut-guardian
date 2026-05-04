# Coconut Guardian

Detect coconut tree diseases from an uploaded image using a TensorFlow/Keras model, with an optional Grad-CAM heatmap visualization.

## Demo

- **Live app (Cloud Run)**: https://coconut-guardian-1079917738080.asia-south1.run.app
- **Demo video**: https://drive.google.com/file/d/1W-ZLAMUAE7zEtc8JUTR4rmr71Kta10aS/view?usp=drivesdk

## Features

- Upload an image and get a predicted disease label + confidence
- Grad-CAM heatmap overlay
- Simple Flask UI

## Tech stack

- **Backend**: Flask
- **ML**: TensorFlow / Keras
- **Image**: Pillow, OpenCV (headless)
- **Deployment**: Docker + Google Cloud Run

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PORT=8080
python .\app.py