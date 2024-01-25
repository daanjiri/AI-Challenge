import cv2
import torch
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
import numpy as np
import io


def detect_face(frame, mtcnn):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    bounding_boxes = bounding_boxes[probs > 0.9]
    return bounding_boxes


def predict_emotion(file_path):
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

    frame_bgr = cv2.imread(file_path)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    bounding_boxes = detect_face(frame, mtcnn)

    model_name = "enet_b0_8_best_afew"
    fer = HSEmotionRecognizer(model_name=model_name, device=device)

    list_inferences = []

    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        face_img = frame[y1:y2, x1:x2, :]
        emotion, _ = fer.predict_emotions(face_img, logits=True)
        list_inferences.append(
            {
                "emotion": emotion,
                "box": [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)],
            }
        )

    # print(list_inferences)
    return list_inferences


# from fer import FER
# import cv2
# from typing import Dict, Union
# from deepface import DeepFace
# from flask import Flask, request, jsonify
# from hsemotion.facial_emotions import HSEmotionRecognizer
# from PIL import Image
# import io

# def detect_face(frame):
#     bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
#     bounding_boxes=bounding_boxes[probs>0.9]
#     return bounding_boxes

# def detect_emotion(path: str) -> Dict[str, Union[str, float]]:
#     img = cv2.imread(path)
#     detector = FER()
#     emotion, score = detector.top_emotion(img)
#     print("emotionnn: ", emotion, score)

#     return {"emotion": emotion}

#     # result = DeepFace.analyze(img_path=path, actions=["emotion"])
#     # print(result)
#     # dominant_emotion = result[0]["dominant_emotion"]
#     # print(result[0]["dominant_emotion"])
#     # return {"emotion": dominant_emotion}


# detect_emotion("./test_images/disgust_demo.jpg")

# deepface, Fer(face emotional recognition)
# https://github.com/av-savchenko/hsemotion
