🚗 Driver Drowsiness Detection using MobileNet
Enhancing Road Safety with Real-Time Deep Learning

This project presents a robust deep learning-based approach for detecting driver drowsiness using a lightweight and efficient MobileNet model. The goal is to reduce road accidents by identifying signs of drowsiness in real time and triggering timely alerts.

🔍 Project Overview
Drowsy driving is a critical concern in road safety. Traditional methods using classical machine learning algorithms like KNN, SVM, and Decision Trees achieved only 75–80% accuracy. This project significantly improves performance using MobileNet, achieving a test accuracy of 95% on a balanced image dataset.

Model: MobileNet (fine-tuned for binary classification)

Accuracy: 95% on validation/test set

Dataset: ~6000 balanced images labeled as Drowsy and Alert

Published in: IEEE (link/publication info can be added here)

Libraries: OpenCV for video processing and face region detection

🧠 Key Features
Real-time driver drowsiness detection via webcam or video stream

Lightweight MobileNet architecture, suitable for edge deployment

OpenCV-powered eye and face region detection

High model performance compared to traditional ML models

Supports easy integration with alert systems (buzzer, LEDs, etc.)

📊 Dataset
Size: ~6000 labeled facial images

Class Distribution: Balanced (Drowsy: 50%, Alert: 50%)

Preprocessing: Resized to 224x224, normalized for MobileNet input

⚙️ Technologies Used
Python

TensorFlow / Keras

OpenCV

NumPy, Pandas

Matplotlib / Seaborn (for visualization)

📈 Results
Model	Accuracy (%)
K-Nearest Neighbors (KNN)	78%
Support Vector Machine (SVM)	80%
Decision Tree	76%
MobileNet (Proposed)	95%

✅ Performance Boost: Achieved a significant improvement of 15-20% in accuracy compared to baseline models.

📝 Publication
Published in: IEEE (10.1109/ICDT63985.2025.10986727)

The paper details the methodology, dataset preparation, model tuning, and comparison with baseline methods.

