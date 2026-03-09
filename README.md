# 🚀 Bulk Watermark Removal Tool (YOLOv11 + LaMA)

An automated, AI-powered tool designed to detect and remove watermarks from images with high precision. Using **YOLOv11** for lightning-fast detection and **LaMA (Large Mask Inpainter)** for seamless texture filling, this tool ensures your images look clean and professional.

---

## 🛠 Features
- **🤖 AI-Powered Detection**: Specifically trained YOLOv11 model to find watermarks.
- **🎨 Deep Texture Filling**: Uses LaMA inpainting to fill removed areas invisibly.
- **⚡ Multi-Threaded**: Process entire folders of images in parallel.
- **📁 Watch Mode**: Real-time processing—drop a file in, and it's cleaned instantly.
- **🔄 Smart Checkpoints**: Automatically skips already processed images.

---

## � 1. Quick Installation (Step-by-Step)

Follow these steps to get the tool running on your system.

### Step 1: Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/Devilmax24-tech/image-watermark-removal.git
cd image-watermark-removal
```

### Step 2: Set Up Virtual Environment (Recommended)
This keeps your Python dependencies organized.
```bash
# Create the environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download the AI Model
You need the trained YOLOv11 weights (`best.pt`) for detection to work. Run this command to download it directly into the `models/` folder:

```bash
# Create the models directory if it doesn't exist
mkdir -p models

# Download the model weights
wget -O models/best.pt https://github.com/Devilmax24-tech/image-watermark-removal/releases/download/v1.0/best.pt
```
> [!NOTE]
> If the `wget` link above is not yet active, please manually place your `best.pt` file into the `models/` directory.

---

## 📂 2. Folder Setup Guide

The tool looks for images in an **Input** folder and saves results in an **Output** folder.

### Default Structure
By default, the tool expects this structure:
```text
image-watermark-removal/
├── data/
│   ├── input/   <-- Place your images here
│   └── output/  <-- Cleaned images will appear here
```

### How to set up your folders:
1. **Create the folders**: The program will create them automatically if they don't exist, but you can create them manually:
   ```bash
   mkdir -p data/input data/output
   ```
2. **Add Images**: Simply copy or move the images you want to clean into `data/input`.
3. **Check Results**: Once the program runs, cleaned images will be moved to `data/output`. Any images that the AI couldn't process will be moved to `data/output/failed`.

---

## 💻 3. How to Use

### A. Run a Single Batch
Process all images currently in your input folder and then stop:
```bash
python main.py --input data/input --output data/output
```

### B. Use "Watch Mode" (Drop & Clean)
Keep the program running. Every time you drop a new image into the input folder, it will be cleaned automatically:
```bash
python main.py --input data/input --output data/output --watch
```

### C. Advanced Options
| Option | Command |
| :--- | :--- |
| **Increase Speed** | `python main.py --workers 8` (Uses 8 parallel threads) |
| **Re-process Everything** | `python main.py --force` (Ignores previous history) |
| **Custom Folders** | `python main.py --input ./my_images --output ./cleaned` |

---

## 🔧 Command Line Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | Path to folder with watermarked images. | `data/input` |
| `--output` | Path to save cleaned images. | `data/output` |
| `--model` | Path to the YOLOv11 `.pt` file. | `models/best.pt` |
| `--workers` | Number of simultaneous image processes. | `4` |
| `--watch` | Continuously watch input folder for new files. | `Off` |
| `--force` | Process all images, even if done before. | `Off` |
| `--interval` | Seconds to wait between checks in watch mode. | `2` |

---

## 📂 Project Architecture
- `main.py`: Entry point for the application.
- `src/detector.py`: AI logic for finding watermarks.
- `src/inpainter.py`: AI logic for removing watermarks.
- `src/processor.py`: Handles multi-threading and folder management.
- `src/utils.py`: Image quality validation (SSIM/PSNR).
- `checkpoint.txt`: History file to avoid re-processing the same image twice.

---

## 📜 License
This project is licensed under the MIT License. Use responsibly.
