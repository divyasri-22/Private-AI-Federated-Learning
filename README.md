# Private AI: Federated Learning Framework

This project implements a privacy-preserving artificial intelligence framework using Federated Learning.
The system avoids centralized data storage by training models across multiple clients.

Human Activity Recognition (HAR) is used as a case study to validate the framework.
The UCI HAR dataset is utilized for training and evaluation.

The model is trained using Federated Averaging (FedAvg) with Logistic Regression.
A global model is created by aggregating locally trained models.

The trained model achieves 94.74% accuracy on test data.
The system is deployed as a web-based prediction application using Streamlit.

The application displays predicted activity, confidence score, and explainable visualizations.
Robustness analysis is included to evaluate prediction stability.

Dataset files are excluded from the repository due to size constraints.
The dataset can be downloaded from the official UCI repository.
