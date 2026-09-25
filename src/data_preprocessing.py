import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

TARGET_CLASSES = [
    'NORMAL',
    'PNEUMONIA'
]

def clahe_preprocessing(img):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    img_uint8 = img.astype('uint8')
    if img_uint8.shape[2] == 3:
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final.astype('float32')
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img_uint8).astype('float32')

def verify_and_count_dataset(dataset_dir):
    """
    Verifies dataset structure and counts images for all 7 disease classes.
    Creates missing disease subdirectories automatically.
    """
    splits = ['train', 'val', 'test']
    counts = {}
    
    print(f"--- Multi-Disease Dataset Verification ({dataset_dir}) ---")
    if os.path.exists(os.path.join(dataset_dir, 'chest_xray', 'train')):
        dataset_dir = os.path.join(dataset_dir, 'chest_xray')

    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        counts[split] = {}
        
        print(f"\n{split.upper()} set:")
        for cls in TARGET_CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            num_images = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            counts[split][cls] = num_images
            print(f"  - {cls:<20}: {num_images} images")
            
    print("--------------------------------------------")
    return counts

def get_data_generators(dataset_dir, image_size=(224, 224), batch_size=32):
    """
    Creates ImageDataGenerators for multi-class training and evaluation.
    Applies data augmentation to training data.
    """
    if os.path.exists(os.path.join(dataset_dir, 'chest_xray', 'train')):
        dataset_dir = os.path.join(dataset_dir, 'chest_xray')

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')

    def has_images(directory):
        if not os.path.exists(directory): return False
        for root, dirs, files in os.walk(directory):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                return True
        return False

    if not has_images(val_dir):
        print("Validation set empty, falling back to test set for validation.")
        val_dir = test_dir

    # Enhanced Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=clahe_preprocessing
    )

    test_val_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=clahe_preprocessing
    )

    print("\nLoading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("Loading Validation Data...")
    val_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("Loading Testing Data...")
    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
