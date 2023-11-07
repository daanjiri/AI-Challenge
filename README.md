# Face-Emotion Recognition System

This project is a real-time facial recognition and emotion detection system developed in Python using OpenCV.

## Overview

The system processes images, identifies faces, and for each identified face, it recognizes the individual and determines their emotional state. It operates in real-time, providing immediate responses.

## Requirements

- Python 3.7 or later
- OpenCV
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository.
2. Install the required packages.

```bash
pip install -r requirements.txt

```

## Usage

After installing the required packages, you can run the program with the following command:

```bash
python main.py
```

This will start the facial recognition and emotion detection system. The system will process images from your webcam in real-time.

If you want to process a single image, you can pass the image path as a command-line argument:

```bash
python main.py --image_path /path/to/your/image.jpg
```
Replace /path/to/your/image.jpg with the actual path to the image you want to process.

## How to add new worker to recognize

### Step 1: Create a folder with the folder name being the name of the person
### Step 2: Add the person's photo in the folder
### Step 3: Move folder to additional-training-datasets folder

#### Example:

- |database
- ----|additional-training-datasets
- --------|name-person1
- --------|name-person2

### Step 4: Run to add person

````
python train.py --is-add-user=True
