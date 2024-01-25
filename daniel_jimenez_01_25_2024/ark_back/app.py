from model import predict_emotion
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


@app.route("/detect-emotion", methods=["POST"])
def handle_emotion_detection():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        save_directory = os.path.join(os.getcwd(), "uploads")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        file_path = os.path.join(save_directory, file.filename)
        file.save(file_path)

        emotions = predict_emotion(file_path)
        print(emotions[0])
        try:
            results_file_path = os.path.join(os.getcwd(), "emotion_results.txt")
            with open(results_file_path, "a") as results_file:
                results_file.write(f"{file_path}\t")
                results_file.write(f"{emotions[0]['emotion']}\t")
                results_file.write(f"{emotions[0]['box']}\t")
                results_file.write("\n")
        except Exception as e:
            print(f"Error writing to file: {e}")

        return jsonify(emotions[0])


if __name__ == "__main__":
    app.run(debug=True)
