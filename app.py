from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import main

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()  # get data from POST request
    image_b64 = data['image']  # get base64 encoded image from JSON
    image_data = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # convert image from base64 to OpenCV format

    result = main.main_function(img)  # call your main function with the image
    return jsonify(result)  # return result as JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)