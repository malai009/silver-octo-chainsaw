#  Computer Vision Experiments with OpenCV & YOLO

This repository contains a collection of **computer vision experiments and mini-projects** implemented as part of my learning and exploration of image processing, object detection, and deep learning–based vision techniques.

The scripts cover both **classical OpenCV techniques** and **deep learning approaches using YOLO**, focusing on building intuition and practical CV pipelines.

##  Repository Overview

###  Classical Computer Vision (OpenCV)

* **EdgeDetector.py**
  Implements edge detection techniques (Sobel / Canny) to highlight image boundaries and structural features.

* **adaptive_thresholding.py**
  Demonstrates adaptive thresholding for handling uneven lighting conditions in images.

* **color_detection.py**
  Performs color-based object detection using HSV color space and masking techniques.


###  Face Processing

* **face_anonymizer.py**
  Detects faces in an image and anonymizes them using blurring.

* **face_anonymizer_cam.py**
  Real-time face anonymization using webcam video input.


###  Image Classification & Segmentation using YOLO

* **img_classify_yolo.py**
  Image classification using a pretrained YOLO model.

* **img_class_yolo_trained_model.py**
  Image classification using a custom-trained YOLO model.

* **img_seg_yolo.py**
  Image segmentation using YOLO to extract object-level masks.


###  Basic Image Classification

* **Image_classification.py**
  A simple image classification pipeline used to understand the fundamentals of visual feature extraction and classification.


##  Key Concepts Covered

* Image preprocessing and filtering
* Edge detection and thresholding
* Color space transformation (BGR → HSV)
* Face detection and anonymization
* YOLO-based object detection, classification, and segmentation
* Real-time video processing using OpenCV


##  Tech Stack

* Python
* OpenCV
* Ultralytics YOLO
* NumPy


##  How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   ```
2. Install required libraries:

   ```bash
   pip install opencv-python ultralytics numpy
   ```
3. Run any script individually:

   ```bash
   python EdgeDetector.py
   ```

> Some scripts require image/video input paths or a webcam.


##  Purpose of This Repository

* Hands-on learning of computer vision concepts
* Understanding both **traditional** and **deep learning–based** vision methods
* Building reusable CV pipelines for future projects
* Preparation for advanced CV tasks such as object tracking and edge deployment


## 🚀 Future Work

* Integrating detection + tracking pipelines
* Performance optimization for edge devices (Raspberry Pi)
* Object counting and motion analysis
* Deployment-ready CV applications


## 📄 Disclaimer

This repository is intended for **educational and experimental purposes**.
