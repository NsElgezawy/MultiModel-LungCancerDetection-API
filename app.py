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

    prediction, confidence = predict(model_name, [img])

    
    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)
