from pathlib import Path
import random
import numpy as np
import argparse
import time
import os
import torch.backends.cudnn as cudnn
import torch
import cv2

from emotion import detect_emotion, init
from Face_recognition import face_recognition_main, init_face_recognition, get_features
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, set_logging, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

#pytorch
from concurrent.futures import thread
from sqlalchemy import null
from torchvision import transforms
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

isThread = True
flag_recognized = False

def plot_one_box(x, img, color=None, name_label=None, emotion_label=None, line_thickness=None, text_color=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    if name_label == 'UN_KNOWN':
        color = [0, 0, 255]  # Red color
        emotion_label = None  # No emotion label
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if text_color is None:
        text_color = [225, 255, 255]  # Use default text color if none is provided
    tf_ = max(tl - 1, 1)  # font thickness

    # Calculate font scale based on bounding box width
    font_scale = max((c2[0] - c1[0]) / 200.0, 0.4)

    if name_label:
        t_size = cv2.getTextSize(name_label, 0, fontScale=font_scale, thickness=tf_)[0]
        c2_aux = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, (c2_aux[0], c1[1] + t_size[1] + 3), color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, name_label, (c1[0], c1[1] + t_size[1] + 2), 0, font_scale, text_color, thickness=tf_, lineType=cv2.LINE_AA)

    if emotion_label:
        t_size = cv2.getTextSize(emotion_label, 0, fontScale=font_scale, thickness=tf_)[0]
        c1 = c1[0], c2[1] - t_size[1] - 3
        c2 = c1[0] + t_size[0], c2[1]
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, emotion_label, (c1[0], c2[1] - 2), 0, font_scale, text_color, thickness=tf_, lineType=cv2.LINE_AA)

def recognition(det,im0):

    global isThread
    
    recognized_faces = []
    
    flag_recognized = False
    for *xyxy, conf, cls in reversed(det):

        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        face_image = im0.astype(np.uint8)[int(y1):int(y2), int(x1): int(x2)]
        recognized_face = face_recognition_main(face_image)
        
        
        name = recognized_face.split(":")[0]
        score = float(recognized_face.split(":")[1])

        if score < 0.30:
            recognized_face = "UN_KNOWN"
        else:
            recognized_face = f"{name.split('_')[0].upper()}"

        recognized_faces.append(recognized_face)
    
    isThread = True

    return recognized_faces

def detect(opt):
    # Extracting options
    source, view_img, imgsz, nosave, show_conf, save_path, show_fps = opt.source, not opt.hide_img, opt.img_size, opt.no_save, not opt.hide_conf, opt.output_path, opt.show_fps
    
    # source = ('http://192.168.1.4:8080/video')
    # source = ('C:/Users/Jonathan/Desktop/Arkangel/video.mp4')
    
    # Checking if the source is a webcam, a text file, or a URL
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Creating directories
    create_folder(save_path)

    # Initializing device
    set_logging()
    device = select_device(opt.device)

    # Initializing model_emotion and model_face_recognition
    init(device)
    init_face_recognition(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Loading model for face detection
    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Converting model to half precision if CUDA is available
    if half:
        model.half()  # to FP16

    # Setting Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        

    # Getting names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Defining colors for different emotions
    colors = (
        (128, 0, 32),       # Anger - Red wine
        (238, 130, 238),   # Contempt - Light Purple
        (34, 139, 34),     # Disgust - Dark Green
        (0, 0, 139),       # Fear - Dark Blue
        (255, 255, 0),     # Happy - Bright Yellow
        (128, 128, 128),   # Neutral - Medium Gray
        (0, 191, 255),     # Sad - Light Blue
        (173, 255, 47)     # Surprise - Light Green
    )

    text_colors = (
        (255, 255, 255),  # For dark rectangle colors - White Text
        (0, 0, 0),       # For light rectangle colors - Black Text
        (0, 0, 0),       # For bright rectangle colors - Black Text
        (255, 255, 255),  # For dark/mid-light rectangle colors - White Text
        (0, 0, 0),  # For light/bright rectangle colors - Black Text
        (255, 255, 255),  # For dark/mid-light rectangle colors - White Text
        (0, 0, 0),       # For all colors - Black Text
        (0, 0, 0)   # For dark/mid-light rectangle colors - Black Text
    )
    
    # Record the start time
    t0 = time.time()

    # Loop over each image in the dataset
    for path, img, im0s, vid_cap in dataset:
        # Convert the image to a PyTorch tensor and move it to the device
        img = torch.from_numpy(img).to(device)
        # Convert the image data type to half precision if half is True, otherwise to single precision
        img = img.half() if half else img.float()
        # Normalize the image from 0-255 to 0.0-1.0
        img /= 255.0
        # If the image only has three dimensions, add a batch dimension
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Record the start time of the inference
        t1 = time_synchronized()
        # Run the model on the image and get the predictions
        pred = model(img, augment=opt.augment)[0]

        # Apply Non-Maximum Suppression (NMS) to the predictions
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        # Record the end time of the inference
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # If webcam is used, the batch size is greater than or equal to 1
            if webcam:  
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # Convert the path to a Path object
            s += '%gx%g ' % img.shape[2:]  # Add the image dimensions to the print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Get the normalization gain

            if len(det):
                # Rescale the bounding boxes from the size of the input image to the original image
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print the detection results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # Count the number of detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Add the class and count to the print string

                images = []
                # Loop over the detections in reverse order
                for *xyxy, conf, cls in reversed(det):
                    # Get the coordinates of the bounding box
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    # Append the cropped image within the bounding box to the images list
                    images.append(im0.astype(np.uint8)[int(y1):int(y2), int(x1): int(x2)])

                if images:
                    # Recognize the faces in the images
                    recognized_faces = recognition(det,im0)
                    # Detect the emotions in the images
                    emotions = detect_emotion(images,show_conf)

                # Write the results
                i = 0
                # Loop over the detections in reverse order
                for *xyxy, conf, cls in reversed(det):
                    if view_img or not nosave:  
                        # Get the emotion label and the corresponding color and text color
                        emotion_label = emotions[i][0].split("(")[0]
                        colour = colors[emotions[i][1]]
                        text_colour = text_colors[emotions[i][1]]
                        
                        # Get the recognized face name
                        name_label = recognized_faces[i]

                        i += 1
                        
                        # Call the function for plotting the bounding box and the label
                        plot_one_box(xyxy, im0, name_label=name_label, emotion_label=emotion_label, 
                                     color=colour, line_thickness=opt.line_thickness,text_color=text_colour)

            # Stream results
            if view_img:
                # Resize the image to double its original size
                display_img = cv2.resize(im0, (im0.shape[1]*2,im0.shape[0]*2))

                # # Create a window
                # cv2.namedWindow("Face-Emotion Recognition", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

                # # Set the window to fullscreen
                # cv2.setWindowProperty("Face-Emotion Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                # Display the image in the window
                cv2.imshow("Face-Emotion Recognition", display_img)

                # Wait for 1 millisecond
                cv2.waitKey(1)

            if not nosave:
                # Check the output format by getting the file extension of the save path
                ext = save_path.split(".")[-1]
                if ext in ["mp4","avi"]:
                    # Save the results (image with detections) as a video
                    if vid_path != save_path:  # If it's a new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # Release the previous video writer
                        if vid_cap:  # If it's a video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # If it's a stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                elif ext in ["bmp", "pbm", "pgm", "ppm", "sr", "ras", "jpeg", "jpg", "jpe", "jp2", "tiff", "tif", "png"]:
                    # Save the image in the specified format
                    cv2.imwrite(save_path,im0)
                else:
                    # Save the image in a folder
                    output_path = os.path.join(save_path,os.path.split(path)[1])
                    create_folder(output_path)
                    cv2.imwrite(output_path,im0)

            if show_fps:
                # Calculate and display the frames per second (fps)
                print(f"FPS: {1/(time.time()-t0):.2f}"+" "*5,end="\r")
                t0 = time.time()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="1", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='face confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hide-img', action='store_true', help='hide results')
    save = parser.add_mutually_exclusive_group()
    save.add_argument('--output-path', default="output.mp4", help='save location')
    save.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--show-fps', default=False, action='store_true', help='print fps to console')
    opt = parser.parse_args()
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        detect(opt=opt)
