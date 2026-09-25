import os
import sys
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import traceback

# Add src to python path to import prediction module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from predict import predict_image, DISEASE_KNOWLEDGE_BASE, warmup_model

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/diseases', methods=['GET'])
def get_supported_diseases():
    """Returns the matrix of supported thoracic diseases and their accuracy benchmarks."""
    diseases_list = []
    for key, info in DISEASE_KNOWLEDGE_BASE.items():
        diseases_list.append({
            "id": key,
            "title": info["title"],
            "accuracy": info["accuracy"],
            "description": info["description"]
        })
    return jsonify({
        "success": True,
        "total": len(diseases_list),
        "diseases": diseases_list
    })

@app.route('/predict', methods=['POST'])
def handle_prediction():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file part provided.'}), 400
            
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Execute full multi-disease AI inference
                result = predict_image(filepath)
                
                # Cleanup temporary uploaded image
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                return jsonify(result)
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                print(f"Prediction Error: {e}")
                traceback.print_exc()
                return jsonify({'error': f'AI Diagnostic processing failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Unsupported file format. Please upload PNG, JPG, or JPEG images.'}), 400

    except Exception as e:
        print(f"Server Error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    warmup_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
