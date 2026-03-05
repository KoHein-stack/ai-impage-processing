# AI Image Processing Suite

Streamlit app for:
- Text extraction (OCR)
- Object segmentation
- Image caption generation

## Project Structure

- `ui.py`: Streamlit user interface
- `model.py`: model loading and caching
- `ocr.py`: OCR processing
- `bg_remove.py`: object segmentation/background removal
- `image_captioning.py`: image caption generation
- `requirements.txt`: Python dependencies

## Prerequisites

- Windows PowerShell
- Python 3.13 (your current setup) or another supported Python version

## Setup

```powershell
cd "\ai-impage-processing"

"python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
python -m streamlit run .\ui.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

## Notes

- First launch can take longer because models/assets may download.
- `yolo11n.pt` is used if `yolo11n-seg.pt` is not present.
- For OCR, model files are downloaded automatically when needed.

## License

This project is distributed under the terms in [LICENSE](LICENSE).
