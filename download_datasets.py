import os
import sys

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

DISEASE_CLASSES = [
    'NORMAL',
    'PNEUMONIA'
]

KAGGLE_DATASETS = {
    "Pneumonia Chest X-Ray (Bacterial & Viral)": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
    "COVID-19 Radiography Database": "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database",
    "Tuberculosis Chest X-Ray Database": "https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset",
    "NIH ChestX-ray14 (Pneumothorax & Effusion)": "https://www.kaggle.com/datasets/nih-chest-xrays/sample"
}

def create_folder_structure():
    print("Creating disease dataset directory structure...")
    for split in ['train', 'val', 'test']:
        for cls in DISEASE_CLASSES:
            folder_path = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(folder_path, exist_ok=True)
            num_files = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  [+] dataset/{split}/{cls}/ ({num_files} images)")

def show_download_guide():
    print("\n" + "=" * 75)
    print("     HOW TO DOWNLOAD CHEST X-RAY DISEASE DATASETS FROM GOOGLE & KAGGLE")
    print("=" * 75)
    print("\nMethod 1: Direct Kaggle / Google Search Links")
    for name, url in KAGGLE_DATASETS.items():
        print(f"  • {name:<45} : {url}")
        
    print("\nMethod 2: Automated KaggleHub Python Download")
    print("  Run the following commands in your terminal:")
    print("    pip install kagglehub")
    print("    python -c \"import kagglehub; path = kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia'); print(path)\"")
    
    print("\nMethod 3: How to place downloaded images into your project:")
    print("  Copy downloaded X-ray image files (.jpg / .png) into their matching folder:")
    print("    Pneumonia_Project/dataset/train/NORMAL/")
    print("    Pneumonia_Project/dataset/train/BACTERIAL_PNEUMONIA/")
    print("    Pneumonia_Project/dataset/train/VIRAL_PNEUMONIA/")
    print("    Pneumonia_Project/dataset/train/TUBERCULOSIS/")
    print("    Pneumonia_Project/dataset/train/COVID19/")
    print("    Pneumonia_Project/dataset/train/PNEUMOTHORAX/")
    print("    Pneumonia_Project/dataset/train/PLEURAL_EFFUSION/")
    print("=" * 75)

def generate_sample_training_data():
    """Generates synthetic initial sample images for testing model training."""
    try:
        from PIL import Image
        import numpy as np
        
        print("\nGenerating sample images for training test...")
        for split in ['train', 'val', 'test']:
            count = 15 if split == 'train' else 5
            for cls in DISEASE_CLASSES:
                folder = os.path.join(DATASET_DIR, split, cls)
                for i in range(count):
                    file_path = os.path.join(folder, f"sample_{cls}_{i+1}.jpg")
                    if not os.path.exists(file_path):
                        # Generate 224x224 grayscale radiograph pattern
                        arr = np.random.randint(40, 220, (224, 224, 3), dtype=np.uint8)
                        img = Image.fromarray(arr)
                        img.save(file_path)
        print("Sample training images created successfully!")
    except Exception as e:
        print(f"Sample generation notice: {e}")

if __name__ == '__main__':
    create_folder_structure()
    if '--generate-samples' in sys.argv:
        generate_sample_training_data()
    show_download_guide()
