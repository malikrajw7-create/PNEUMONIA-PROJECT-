import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from data_preprocessing import verify_and_count_dataset, get_data_generators
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

DATASET_DIR = r"C:\Users\LENOVO\Downloads\ChestXRay2017"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.00005

def build_best_cnn_model(input_shape, num_classes=2):
    """
    Builds a state-of-the-art CNN model using Transfer Learning (DenseNet121)
    optimized for medical image classification like Pneumonia.
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    model = Sequential([
        Input(shape=input_shape),
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def main():
    # 1. Verify Dataset
    counts = verify_and_count_dataset(DATASET_DIR)
    total_images = sum(sum(split.values()) for split in counts.values())
    
    if total_images == 0:
        print("\nNotice: No training images found in dataset/ directories.")
        print("Please run 'python download_datasets.py' or add images to dataset/train/<CLASS_NAME>/")
        return

    # 2. Get Data Generators
    train_gen, val_gen, _ = get_data_generators(DATASET_DIR, IMAGE_SIZE, BATCH_SIZE)
    num_classes = len(train_gen.class_indices)

    # 3. Build Model
    input_shape = (*IMAGE_SIZE, 3)
    model = build_best_cnn_model(input_shape, num_classes=num_classes)
    model.summary()

    # 4. Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'pneumonia_custom_cnn.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    # Calculate Class Weights to handle dataset imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("\nClass Weights applied to handle dataset imbalance:", class_weight_dict)

    # 5. Train Model
    print(f"\nStarting State-of-the-Art DenseNet121 Training on {total_images} images across {num_classes} classes...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )
    print("\nTraining completed! High-accuracy model saved to:", model_path)

if __name__ == '__main__':
    main()
