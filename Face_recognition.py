import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from insightface.insight_face import iresnet100
import numpy as np
from PIL import Image

model_face_recognition  = iresnet100()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112), antialias=True),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

def init_face_recognition(device):
    # Initialise model emotions
    global dev
    dev = device

    # Initialise model face recognition 

    weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
    model_face_recognition.load_state_dict(weight)
    model_face_recognition.to(device)
    model_face_recognition.eval()

def get_features(face_image, training = True): 
    # Convert to RGB
    # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(dev)

    # Via model to get feature
    with torch.no_grad():
        if training:
            features_img_face = model_face_recognition(face_image[None, :])[0].cpu().numpy()
        else:
            features_img_face = model_face_recognition(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_features = features_img_face/np.linalg.norm(features_img_face)
    return images_features

def face_recognition_main(face_image):

    global isThread, score, name
    
    # Get feature from face
    query_emb = (get_features(face_image, training=False))
    
    # Read features
    root_fearure_path = "static/feature/face_features.npz"
    data = np.load(root_fearure_path, allow_pickle=True)
    images_names_dataset = data["arr1"]
    features_img_face_dataset = data["arr2"]

    scores = (query_emb @ features_img_face_dataset.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names_dataset[id_min]

    result = f"{name.split('_')[0].upper()}:{score:.2f}"

    return result
