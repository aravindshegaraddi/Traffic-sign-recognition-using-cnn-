import numpy as np
import cv2
from tensorflow import keras

frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.90  # probability threshold
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# LOAD THE TRAINED CNN MODEL

model = keras.models.load_model("model_trained.keras")


# PREPROCESSING FUNCTIONS FOR IMAGE CAPTURED FROM WEBCAM 

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img


# CLASS LABEL MAPPING


classes = [
    'Speed Limit 20 km/h',
    'Speed Limit 30 km/h',
    'Speed Limit 50 km/h',
    'Speed Limit 60 km/h',
    'Speed Limit 70 km/h',
    'Speed Limit 80 km/h',
    'End of Speed Limit 80 km/h',
    'Speed Limit 100 km/h',
    'Speed Limit 120 km/h',
    'No passing',
    'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection',
    'Priority road',
    'Yield',
    'Stop',
    'No vehicles',
    'Vehicles over 3.5 metric tons prohibited',
    'No entry',
    'General caution',
    'Dangerous curve to the left',
    'Dangerous curve to the right',
    'Double curve',
    'Bumpy road',
    'Slippery road',
    'Road narrows on the right',
    'Road work',
    'Traffic signals',
    'Pedestrians',
    'Children crossing',
    'Bicycles crossing',
    'Beware of ice/snow',
    'Wild animals crossing',
    'End of all speed and passing limits',
    'Turn right ahead',
    'Turn left ahead',
    'Ahead only',
    'Go straight or right',
    'Go straight or left',
    'Keep right',
    'Keep left',
    'Roundabout mandatory',
    'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons'
]

def getClassName(classNo):
    if classNo < len(classes):
        return classes[classNo]
    else:
        return "No Vehicle"

# MAIN LOOP


while True:
    success, imgOriginal = cap.read()
    if not success:
        break

    # PREPROCESS IMAGE
    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)

    # RESHAPE FOR PREDICTION
    img_input = img.reshape(1, 32, 32, 1)

    # PREDICT
    predictions = model.predict(img_input)
    classIndex = int(np.argmax(predictions))
    probabilityValue = float(np.max(predictions))

    # DISPLAY RESULTS
    cv2.putText(imgOriginal, "CLASS:", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "PROBABILITY:", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    if probabilityValue > threshold:
        className = getClassName(classIndex)
        cv2.putText(imgOriginal, f"{classIndex} - {className}", (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, f"{round(probabilityValue * 100, 2)}%", (220, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOriginal)

    # QUIT WITH 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
