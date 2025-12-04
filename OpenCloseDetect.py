#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# In[32]:


# Define the path to dataset folders
open_eyes_folder = '/Users/train/Open_Eyes'
closed_eyes_folder = '/Users/train/Closed_Eyes'


# In[33]:


# Define the input image dimensions
img_width, img_height = 64, 64


# In[34]:


# Load and preprocess the images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
    return images

open_eyes_images = load_images_from_folder(open_eyes_folder)
closed_eyes_images = load_images_from_folder(closed_eyes_folder)


# In[35]:


# Create labels for the images (1 for open eyes, 0 for closed eyes)
open_eyes_labels = np.ones(len(open_eyes_images))
closed_eyes_labels = np.zeros(len(closed_eyes_images))


# In[36]:


# Concatenate the images and labels
images = open_eyes_images + closed_eyes_images
labels = np.concatenate([open_eyes_labels, closed_eyes_labels])


# In[37]:


# Convert the images to numpy arrays and normalize the pixel values
X = np.array(images) / 255.0
y = np.array(labels)


# In[38]:


# Reshape the input data
X = X.reshape(-1, img_width, img_height, 1)


# In[39]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[41]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[42]:


# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))


# In[43]:


# Load and preprocess the test images
test_folder = '/Users/test'
test_images = load_images_from_folder(test_folder)
test_images = np.array(test_images) / 255.0
test_images = test_images.reshape(-1, img_width, img_height, 1)


# In[44]:


# Predict on the test images
predictions = model.predict(test_images)
predicted_labels = ['Open' if pred > 0.5 else 'Closed' for pred in predictions]


# In[45]:


# Show and print the predictions for 20 test images
for i in range(23):
    image_path = os.listdir(test_folder)[i]
    image = cv2.imread(os.path.join(test_folder, image_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_width, img_height))
    
    prediction = predicted_labels[i]
    
    cv2.imshow('Test Image', image)
    print(f"Image: {image_path} - Prediction: {prediction}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




