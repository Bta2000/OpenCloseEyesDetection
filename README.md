# Open / Closed Eyes Detection

A simple computer vision project that uses a Convolutional Neural Network (CNN) to classify eye images as **Open** or **Closed**.

## About the Project

This project was developed as a practical introduction to image processing and deep learning.

The model learns to distinguish between open and closed eyes using labeled grayscale images. After training, the model can be used to classify new eye images as either `Open` or `Closed`.

## Technologies

* Python
* OpenCV
* NumPy
* Scikit-learn
* Keras / TensorFlow
* Convolutional Neural Network (CNN)

## How It Works

The project follows these main steps:

1. Load images from the `Open_Eyes` and `Closed_Eyes` dataset folders.
2. Convert images to grayscale.
3. Resize images to `64 × 64` pixels.
4. Normalize pixel values to a range between 0 and 1.
5. Assign labels:

   * `1` → Open eyes
   * `0` → Closed eyes
6. Split the dataset into training and testing sets.
7. Build and train a CNN model.
8. Use the trained model to predict whether new images contain open or closed eyes.
9. Display the test images together with their predicted labels.

## CNN Architecture

The model consists of:

* Convolutional layer with 32 filters
* Max Pooling layer
* Convolutional layer with 64 filters
* Max Pooling layer
* Flatten layer
* Fully connected layer with 64 neurons
* Output layer with a sigmoid activation function

The model is trained using the **Adam optimizer** and **binary cross-entropy loss**.

## Dataset Structure

The project expects the training data to be organized into two folders:

```text
train/
├── Open_Eyes/
└── Closed_Eyes/
```

Test images are placed in:

```text
test/
```

## Project Structure

```text
OpenCloseEyesDetection/
│
├── OpenCloseDetect.py
├── test/
└── README.md
```

> The training dataset is not included in this repository. The folder paths in `OpenCloseDetect.py` should be updated according to the location of your local dataset.

## Running the Project

Install the required Python packages:

```bash
pip install opencv-python numpy scikit-learn tensorflow
```

Then update the dataset paths in `OpenCloseDetect.py`:

```python
open_eyes_folder = '/path/to/train/Open_Eyes'
closed_eyes_folder = '/path/to/train/Closed_Eyes'
test_folder = '/path/to/test'
```

Run the script:

```bash
python OpenCloseDetect.py
```

The model will be trained and then used to classify the images in the `test` folder.

## Example Output

For each test image, the predicted result is printed in the following format:

```text
Image: example.jpg - Prediction: Open
```

The corresponding test image is also displayed using OpenCV.

## Purpose

This project demonstrates a basic end-to-end workflow for image classification:

**Image preprocessing → Dataset preparation → CNN training → Image classification**

## Future Improvements

Possible improvements include:

* Real-time eye detection using a webcam
* Face and eye detection before classification
* Data augmentation
* Model evaluation with precision, recall, and F1-score
* Saving and loading the trained model instead of retraining it every time
* Improving the dataset and model architecture

## Author

**Bita Shahani**

GitHub: [Bta2000](https://github.com/Bta2000)
