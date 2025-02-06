# Emotion Detection Using CNN

## Overview
This project is an emotion detection model built using a Convolutional Neural Network (CNN) in TensorFlow/Keras. The model classifies facial expressions into seven different categories using the "Emotion Detection FER" dataset. It is trained on images of various emotions and can predict emotions on new images.

## Dataset
The dataset is structured with images stored in directories corresponding to different emotions. It consists of training and test images located at:
- **Training data:** `/kaggle/input/emotion-detection-fer/train`
- **Test data:** `/kaggle/input/emotion-detection-fer/test`

## Model Architecture
The CNN model is built using the following layers:
1. **Convolutional Layer (32 filters, kernel size 3x3, ReLU activation)**
2. **Max Pooling Layer (pool size 2x2, stride 2)**
3. **Convolutional Layer (32 filters, kernel size 3x3, ReLU activation)**
4. **Max Pooling Layer (pool size 2x2, stride 2)**
5. **Flattening Layer**
6. **Fully Connected Layer (128 neurons, ReLU activation)**
7. **Output Layer (7 neurons, softmax activation)**

## Data Preprocessing
The images are preprocessed using `ImageDataGenerator` with the following transformations:
- Rescaling pixel values (1./255)
- Applying random shear transformations
- Applying zoom transformations
- Horizontal flipping

## Model Compilation and Training
The model is compiled using:
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Evaluation Metric:** Accuracy

The model is trained for **25 epochs** using:
```python
cnn.fit(x=train_set, validation_data=test_set, epochs=25)
```

## Prediction on Test Image
The model is used to predict emotions on new images by:
1. Loading the test image
2. Preprocessing (resizing, converting to array, normalizing)
3. Expanding dimensions to fit the input shape
4. Predicting the class
5. Displaying the image with the predicted emotion

Example prediction:
```python
prediction = cnn.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
class_names = list(train_set.class_indices.keys())
predicted_class_name = class_names[predicted_class[0]]
```

## Dependencies
To run this project, install the following dependencies:
```bash
pip install tensorflow numpy pandas matplotlib
```

## Results and Visualization
The model predictions can be visualized using Matplotlib:
```python
plt.imshow(test_image)
plt.title(f"Predicted Class: {predicted_class_name}")
plt.show()
```

## Future Improvements
- Enhance model performance with deeper architectures
- Use data augmentation techniques to improve generalization
- Fine-tune hyperparameters for better accuracy


