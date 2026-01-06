from flask import Flask, request, jsonify, render_template
from inference import predict
from PIL import Image

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")  # HTML لازم يكون في templates/index.html


@app.route('/predict', methods=['POST'])
def predict_route():
    
    file = request.files.get('images')
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(file).convert('RGB')

    model_name = request.form.get('model_name')
    if not model_name:
        return jsonify({"error": "No model selected"}), 400

    prediction, confidence, uncertainty = predict(model_name, [img])

    
    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "uncertainty": uncertainty
    })

from inference import compute_gradcam, predict # Ensure compute_gradcam is imported

@app.route('/gradcam', methods=['POST'])
def gradcam_endpoint():
    if 'images' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['images']
    img_bytes = Image.open(file.stream).convert('RGB')
    
    try:
        # Call the new function in inference.py
        base64_img = compute_gradcam([img_bytes])
        return jsonify({'image': base64_img})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
