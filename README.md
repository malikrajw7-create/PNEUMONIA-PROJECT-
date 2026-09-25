# AI-Based Pneumonia Detection from Chest X-Ray Images Using CNN

**AI-Based Early Detection of Lung Diseases Using Chest X-Ray Images and Deep Learning**

## Project Description
A deep learning system that takes a chest X-ray image as input and predicts whether the image is NORMAL or indicates PNEUMONIA. It includes a custom CNN architecture, a transfer learning alternative (MobileNetV2), and a responsive Flask web application for inference.

## Features
- Complete ML pipeline (data loading, preprocessing, model training, evaluation).
- Baseline CNN architecture and MobileNetV2 for transfer learning comparison.
- Professional Flask Web Application with drag-and-drop file upload.
- Detailed Jupyter notebook for educational purposes.

## Technologies
- **Python 3.11+**, **TensorFlow/Keras**, **Flask**, **Scikit-learn**, **Matplotlib**, **Seaborn**, **OpenCV**

## Project Structure
- `dataset/`: Contains `train/`, `val/`, `test/` data splits (NORMAL/PNEUMONIA).
- `notebooks/`: Contains `pneumonia_detection.ipynb` step-by-step guide.
- `src/`: Training, evaluation, and data processing scripts.
- `model/`: Directory for saved trained models (`.keras`).
- `app/`: Flask web application and UI assets.
- `uploads/`: Temporary directory for uploaded images.
- `results/`: Contains evaluation graphs and reports.

## Installation
1. Clone the repository and navigate to the project directory.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup
Download the Mendeley Pneumonia Chest X-Ray dataset and place the images in the corresponding folders:
`dataset/train/NORMAL/`, `dataset/train/PNEUMONIA/`, etc.

## Training Instructions
To train the baseline CNN:
```bash
python src/train.py
```

To train the Transfer Learning model (MobileNetV2):
```bash
python src/train_transfer.py
```

To evaluate the models on the test set:
```bash
python src/evaluate.py
```

## Running Flask Web Application
Start the server:
```bash
python app/app.py
```
Then open `http://127.0.0.1:5000` in your web browser.

## Limitations & Medical Disclaimer
**Educational/research use only. This system is not a substitute for professional medical diagnosis.** AI predictions may contain errors and should not be used to make clinical decisions without a human expert.
