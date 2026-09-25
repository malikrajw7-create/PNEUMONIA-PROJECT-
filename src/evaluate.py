import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import get_data_generators

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

def evaluate_model(model_name, image_size=(224, 224)):
    model_path = os.path.join(MODEL_DIR, f'{model_name}.keras')
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    print(f"Loading model: {model_name}...")
    model = tf.keras.models.load_model(model_path)

    print("Loading test data...")
    _, _, test_gen = get_data_generators(DATASET_DIR, image_size=image_size, batch_size=32)

    if test_gen.samples == 0:
        print("Error: No test data found.")
        return

    print("Evaluating model...")
    results = model.evaluate(test_gen, verbose=1)
    
    # Generate predictions
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Handle binary ROC-AUC
    # Probability of the positive class (PNEUMONIA) is predictions[:, 1]
    prob_pos = predictions[:, 1]
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = "N/A"
        
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, prob_pos)
    except ValueError:
        roc_auc = "N/A"

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"Specificity: {specificity:.4f}" if isinstance(specificity, float) else f"Specificity: {specificity}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}" if isinstance(roc_auc, float) else f"ROC-AUC:     {roc_auc}")
    
    report = classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA'])
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(RESULTS_DIR, f'evaluation_{model_name}.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n\n")
        f.write(f"Accuracy:    {acc:.4f}\n")
        f.write(f"Precision:   {prec:.4f}\n")
        f.write(f"Recall:      {rec:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n" if isinstance(specificity, float) else f"Specificity: {specificity}\n")
        f.write(f"F1-Score:    {f1:.4f}\n")
        f.write(f"ROC-AUC:     {roc_auc:.4f}\n\n" if isinstance(roc_auc, float) else f"ROC-AUC:     {roc_auc}\n\n")
        f.write(report)

    # 2. Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}.png'))
    plt.close()

    print(f"Evaluation complete. Results saved in {RESULTS_DIR}")

if __name__ == '__main__':
    print("Evaluating Custom CNN Model...")
    evaluate_model('pneumonia_custom_cnn', image_size=(224, 224))
