from flask import Flask, request, jsonify
from inference import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_api():
    model_name = request.form.get("model")
    images = request.files.getlist("images")

    result = predict(model_name, images)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
