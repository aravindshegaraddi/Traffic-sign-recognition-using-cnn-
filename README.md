# Project RTS - Real-Time Traffic Sign Detection Using CNN

# Overview
<p align="justify">
Project RTS is an advanced real-time traffic sign detection and recognition system designed to enhance road safety and support intelligent transportation solutions. Leveraging deep learning and computer vision, the system identifies and classifies traffic signs from live video feeds, providing instant feedback crucial for driver assistance systems and autonomous vehicles. The model is trained on diverse datasets of Indian and German traffic signs to ensure robust performance across various conditions.
</p>

# Abstract
<p align="justify">
Project RTS implements a real-time pipeline for detecting and classifying traffic signs using a custom Convolutional Neural Network (CNN) architecture. Trained on approximately 35,000 labeled images covering 43 classes, the system processes live video streams captured from a webcam, performing image preprocessing, prediction, and visualization in real time. It overlays recognized traffic signs and confidence scores directly onto the video feed, delivering an efficient solution suitable for deployment on standard computing hardware without GPU acceleration.
</p>

# Table of Contents
- [Demo Photos](#demo-photos)
- [Libraries](#libraries)
- [Block Diagram](#block-diagram)
- [Code Base](#code-base)
- [Technologies Used](#technologies-used)
- [Result](#result)
- [Conclusion](#conclusion)

## Demo Photos

<p align="center">
  <img width="661" height="506" alt="Screenshot 2025-04-27 180606" src="https://github.com/user-attachments/assets/64001106-f329-41bc-85d2-ebc5db2457fc" />
</p>

# Libraries
Libraries Already Developed/Utilized

| Libraries | Description |
| :---         | :---      |
| CNN Model | Custom architecture trained for traffic sign classification |
| OpenCV | Handles video capture, image processing, and visualization |
| NumPy | Used for numerical operations and array manipulation |
| Pandas | Used for data handling and analysis |
| cvzone | Used for simplifying computer vision tasks and overlays |
| scikit-learn | Used for preprocessing and model evaluation utilities |
| TensorFlow/Keras | Deep learning framework used for model development |


**Dataset Details:**

- Total images: ~35,000
- Number of classes: 43
- Classes:
    - 0: Speed limit (20km/h)
    - 1: Speed limit (30km/h)
    - 2: Speed limit (50km/h)
    - 3: Speed limit (60km/h)
    - 4: Speed limit (70km/h)
    - 5: Speed limit (80km/h)
    - 6: End of speed limit (80km/h)
    - 7: Speed limit (100km/h)
    - 8: Speed limit (120km/h)
    - 9: No passing
    - 10: No passing for vehicles over 3.5 metric tons
    - 11: Right-of-way at the next intersection
    - 12: Priority road
    - 13: Yield
    - 14: Stop
    - 15: No vehicles
    - 16: Vehicles over 3.5 metric tons prohibited
    - 17: No entry
    - 18: General caution
    - 19: Dangerous curve to the left
    - 20: Dangerous curve to the right
    - 21: Double curve
    - 22: Bumpy road
    - 23: Slippery road
    - 24: Road narrows on the right
    - 25: Road work
    - 26: Traffic signals
    - 27: Pedestrians
    - 28: Children crossing
    - 29: Bicycles crossing
    - 30: Beware of ice/snow
    - 31: Wild animals crossing
    - 32: End of all speed and passing limits
    - 33: Turn right ahead
    - 34: Turn left ahead
    - 35: Ahead only
    - 36: Go straight or right
    - 37: Go straight or left
    - 38: Keep right
    - 39: Keep left
    - 40: Roundabout mandatory
    - 41: End of no passing
    - 42: End of no passing by vehicles over 3.5 metric tons

<p align="center">
Dataset 
<p align="center">
<img width="375" height="200" alt="image" src="https://github.com/user-attachments/assets/7c0dc381-f15a-4cf4-a2fb-1d2e76e18ff4" />
</p>

# Block Diagram
<p align="center">
System Block Diagram

<img width="875" height="545" alt="Screenshot 2025-05-21 164759" src="https://github.com/user-attachments/assets/f0f1a23b-1b0c-4acc-beba-cbd44d1f2107" />

# Code Base

- CNN Model Architecture and Training Code
- Real-Time Video Capture and Processing Code

# Technologies Used
1. Python: Core language for data processing and model development.
2. OpenCV: Used for video streaming, frame manipulation, and drawing overlays.
3. TensorFlow & Keras: Deep learning frameworks for CNN design and training.
4. NumPy, Pandas: Used for data handling and analysis.
5. Matplotlib: For visualizing model performance and dataset distribution.

# Result
Project RTS successfully demonstrates high-performance real-time detection of traffic signs:

- Achieved ~90% classification accuracy on 43 Indian and German traffic sign classes.
- Processes video at near real-time speeds (~30 FPS) on standard laptops without GPU acceleration.
- Robust to varied lighting, angles, and occlusion due to comprehensive data augmentation during training.
- Clear overlay of detected sign labels and confidence scores on live video, enabling seamless integration into ADAS systems.

# Conclusion
<p align="justify">
Project RTS represents a significant step toward safer and smarter transportation by enabling real-time traffic sign recognition using deep learning. The system is lightweight, scalable, and can be integrated into ADAS frameworks, contributing to accident prevention and improved traffic management. Future improvements may include deployment on embedded platforms, integration with other sensor data, and exploration of advanced detection architectures like YOLO for even faster inference.
<p align="justify">
The project showcases the power of combining computer vision with deep learning to solve critical real-world problems in the field of intelligent transportation systems.
