import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# RANDOM SETTING UP

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# PARAMETERS FOR MODEL TRAINING 

path = "Data"  # folder with all the class folders
labelFile = 'labels.csv'
batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 30
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

# LOAD IMAGES

images = []
classNo = []

myList = sorted(os.listdir(path))
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")

for count, folder in enumerate(myList):
    folder_path = os.path.join(path, folder)
    myPicList = os.listdir(folder_path)
    for y in myPicList:
        curImg_path = os.path.join(folder_path, y)
        curImg = cv2.imread(curImg_path)
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")

images = np.array(images)
classNo = np.array(classNo)

# SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    images, classNo, test_size=testRatio, random_state=42, stratify=classNo
)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=validationRatio, random_state=42, stratify=y_train
)

print("\nData Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_validation.shape, y_validation.shape)
print("Test:", X_test.shape, y_test.shape)

# LOAD CLASSESS FROM LABELS

data = pd.read_csv(labelFile)
print("Labels CSV shape:", data.shape)

# DISPLAY SAMPLES FROM DATASET

num_of_samples = []
cols = 5

fig, axs = plt.subplots(nrows=noOfClasses, ncols=cols, figsize=(5 * cols, 3 * noOfClasses))

for j, row in data.iterrows():
    x_selected = X_train[y_train == j]
    if len(x_selected) == 0:
        for i in range(cols):
            axs[j, i].axis("off")
        num_of_samples.append(0)
        continue

    for i in range(cols):
        idx = random.randint(0, len(x_selected) - 1)
        axs[j, i].imshow(cv2.cvtColor(x_selected[idx], cv2.COLOR_BGR2RGB))
        axs[j, i].axis("off")
        if i == 2:
            axs[j, i].set_title(f"{j} - {row['Name']}")
    num_of_samples.append(len(x_selected))

plt.tight_layout()
plt.show()

print("Samples per class:", num_of_samples)

plt.figure(figsize=(12, 4))
plt.bar(range(0, noOfClasses), num_of_samples)
plt.title("Distribution of training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.tight_layout()
plt.show()

# PREPROCESSING THE DATASET

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

X_train = np.array([preprocessing(img) for img in X_train])
X_validation = np.array([preprocessing(img) for img in X_validation])
X_test = np.array([preprocessing(img) for img in X_test])

plt.imshow(X_train[random.randint(0, len(X_train)-1)], cmap="gray")
plt.title("Example preprocessed image")
plt.axis("off")
plt.show()

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# AUGMENTATION FOR BETTER TRAINING

dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)

batches = dataGen.flow(X_train, y_train, batch_size=15)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(20, 3))
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]), cmap='gray')
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# ONE HOT ENCODING

y_train = to_categorical(y_train, num_classes=noOfClasses)
y_validation = to_categorical(y_validation, num_classes=noOfClasses)
y_test = to_categorical(y_test, num_classes=noOfClasses)

# CNN MODEL 

def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500

    model = Sequential()
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu',
                     input_shape=(imageDimensions[0], imageDimensions[1], 1)))
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
model.summary()

# TRAIN THE MODEL

history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=True
)

# PLOT TRAINING LOSS AND VALIDATION 
# PLOT TRAINING ACCURACY AND VALUATION 

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()


# EVALUATION OF THE TRAINED MODEL 

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# SAVE CNN MODEL

model.save("model_trained.keras")
print("Model saved as model_trained.keras.")
