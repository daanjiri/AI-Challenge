import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image
from repvgg import create_RepVGG_A0

# Load model
model_emotions = create_RepVGG_A0(deploy=True)

# 8 Emotions
emotions = ("Anger","Contempt","Disgust","Fear","Happy","Neutral","Sad","Surprise")


def init(device):
    # Initialise model
    global dev
    dev = device
    model_emotions.to(device)
    model_emotions.load_state_dict(torch.load("weights/repvgg.pth"))

    # Save to eval
    cudnn.benchmark = True
    model_emotions.eval()

def detect_emotion(images,conf=True):
    with torch.no_grad():
        # Normalise and transform images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        x = torch.stack([transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])(Image.fromarray(image)) for image in images])
        # Feed through the model
        pred_emotions = model_emotions(x.to(dev))

        result_emotions = []
        for i in range(pred_emotions.size()[0]):
            # Add emotion to result
            emotion = (max(pred_emotions[i]) == pred_emotions[i]).nonzero().item()
            # Add appropriate label if required
            result_emotions.append([f"{emotions[emotion]}{f' ({100*pred_emotions[i][emotion].item():.1f}%)' if conf else ''}",emotion])
    return result_emotions