# Bulk Watermark Removal Tool (YOLOv11 + LaMA)

An automated tool to detect and remove watermarks from images using AI detection (YOLOv11) and texture-aware inpainting (LaMA). This version is optimized for local processing with folder-drop capabilities.

---

## 🛠 Features
- **🤖 AI-Powered Detection**: Uses YOLOv11 to find watermarks with high precision.
- **🎨 Deep Texture Filling**: Uses LaMA (Large Mask Inpainter) for invisible watermark removal.
- **⚡ Super Speed**: Multi-threaded processing handles multiple images simultaneously.
- **📁 Watch Mode**: Automatically processes images the moment you drop them into a folder.
- **🔄 Force Mode**: Re-process images that were previously skipped.

---

## 🚀 Setup Guide (Step-by-Step)

Follow these steps to set up the tool on any PC (Windows, Linux, or Mac).

### 1. Prerequisites
- **Python 3.9+** installed on your system.
- **Git** installed on your system.

### 2. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/Devilmax24-tech/image-watermark-removal.git
cd image-watermark-removal
```

### 3. Create a Virtual Environment (Recommended)
Keep your system clean by using a virtual environment:
```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Linux/Mac)
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Add the AI Model
1. Browse to the `models/` folder in the project directory.
2. Place your trained YOLOv11 weights file named `best.pt` inside the `models/` folder.
   *(Folder Path: `models/best.pt`)*

---

## 💻 How to Use

### A. Simple Batch Processing
Process all images in a folder and stop:
```bash
python main.py --input path/to/input_folder --output path/to/output_folder
```

### B. Auto-Watch Mode (Drop Folder)
Drop images into a folder and they will be cleaned automatically while the program runs:
```bash
python main.py --input /path/to/desktop/folder --output /path/to/clean_folder --watch
```

### C. Force Re-Processing
Process everything in the folder, even if they were cleaned before:
```bash
python main.py --input path/to/input_folder --output path/to/output_folder --force
```

---

## 🔧 Command Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | Path to the folder containing images to clean. | `data/input` |
| `--output` | Path to the folder to save results. | `data/output` |
| `--workers` | Number of images to process in parallel. | `4` |
| `--watch` | Keep the tool running and watch for new files. | `Off` |
| `--force` | Ignore history and re-clean everything. | `Off` |
| `--interval` | Seconds to wait between checks in watch mode. | `2` |

---

## 📂 Project Structure
- `main.py`: The entry point for running the tool.
- `src/detector.py`: YOLOv11 watermark detection logic.
- `src/inpainter.py`: LaMA inpainting AI wrapper.
- `src/processor.py`: Orchestration, multi-threading, and checkpoint logic.
- `src/utils.py`: SSIM/PSNR image quality metrics.
- `checkpoint.txt`: Automatically keeps track of which images are already done.
