import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# PARAMETERS AND CONFIGURATION
# ==========================================
PATH = "Images"
# Increased resolution to capture the entire scene (full-frame)
IMAGE_DIMENSIONS = (128, 128, 3) 

BATCH_SIZE_VAL = 32     
EPOCHS_VAL = 15         
TEST_RATIO = 0.2        
VAL_RATIO = 0.2         

# ==========================================
# 1. DATASET LOADING
# ==========================================
count = 0
images = []
classNo = []

# Directory check
if not os.path.exists(PATH):
    print(f"Error: Directory '{PATH}' not found.")
    exit()

myList = os.listdir(PATH)
noOfClasses = len(myList)
print(f"Detected classes: {noOfClasses}")
print("Starting loading and resizing...")

for x in range(0, noOfClasses):
    folder_path = os.path.join(PATH, str(count))
    
    if not os.path.exists(folder_path):
        print(f"Warning: Directory {folder_path} does not exist.")
        count += 1
        continue
        
    myPicList = os.listdir(folder_path)
    
    for y in myPicList:
        try:
            img_path = os.path.join(folder_path, y)
            curImg = cv2.imread(img_path)
            
            if curImg is not None:
                # Resizing to network input size
                curImg = cv2.resize(curImg, (IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]))
                images.append(curImg)
                classNo.append(count)
        except Exception as e:
            print(f"Failed to read {y}: {e}")
            
    print(f" -> Class {count} processed.")
    count += 1

images = np.array(images)
classNo = np.array(classNo)

print(f"\nTotal images: {len(images)}")
if len(images) == 0:
    print("Error: Empty dataset.")
    exit()

# ==========================================
# 2. DATA SPLIT
# ==========================================
# The variables y_train and y_test represent the labels for the training and test images, respectively.
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=TEST_RATIO)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VAL_RATIO)

# ==========================================
# 3. PREPROCESSING
# ==========================================
def preprocessing(img):
    # Conversion BGR -> Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogram equalization
    img = cv2.equalizeHist(img)
    # Normalization (0 to 1)
    img = img / 255.0
    return img

print("Applying preprocessing...")
X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# Reshape to single-channel (grayscale) format compatible with Keras
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# ==========================================
# 4. DATA AUGMENTATION
# ==========================================
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    # Tilt the image
    shear_range=0.1,
    rotation_range=10
)
# Fit the generator to training data
dataGen.fit(X_train)

# One-hot encoding of labels
y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# ==========================================
# 5. MODEL ARCHITECTURE
# ==========================================
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500

    model = Sequential()
    
    # Convolutional Block 1
    model.add(Conv2D(no_Of_Filters, size_of_Filter, input_shape=(IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1], 1), activation='relu'))
    model.add(Conv2D(no_Of_Filters, size_of_Filter, activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    
    # Convolutional Block 2
    model.add(Conv2D(no_Of_Filters // 2, (3, 3), activation='relu'))
    model.add(Conv2D(no_Of_Filters // 2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    
    # Convolutional Block 3 (additional for larger inputs)
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    
    model.add(Dropout(0.5)) 

    # Dense layers (fully connected)
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax')) 

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

# ==========================================
# 6. TRAINING
# ==========================================
print("\nStarting training...")
# Dynamic computation of steps per epoch
steps_per_epoch_calc = len(X_train) // BATCH_SIZE_VAL
if steps_per_epoch_calc == 0:
    steps_per_epoch_calc = 1

history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=BATCH_SIZE_VAL),
    steps_per_epoch=steps_per_epoch_calc,
    epochs=EPOCHS_VAL,
    validation_data=(X_validation, y_validation),
    shuffle=True
)

# ==========================================
# 7. METRICS AND MODEL SAVING
# ==========================================
# Loss curve
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')

# Accuracy curve
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.show()

print("Saving model...")
model.save('gesture_model.h5')
print("Model saved to 'gesture_model.h5'")