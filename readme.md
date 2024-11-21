# Breast Cancer Prediction App

## Overview

This project is a Streamlit application designed to predict whether a tumor is **benign** or **malignant** based on various tumor features. The app uses a machine learning model trained on the well-known **Breast Cancer Wisconsin dataset** from the UCI Machine Learning Repository. The app provides a simple interface where users can input tumor feature values and receive a prediction along with the associated probability of malignancy.

## Key Features of the App

- **Probability Display**: The app not only predicts whether the tumor is benign or malignant but also provides the probability associated with the prediction. This allows users to see how confident the model is in its decision.The format is [probability of malignant, probabilty of benign].
- **Selected Features for Prediction**: The app performs feature selection using `SelectKBest` to choose the most relevant features for the prediction task. This improves the model’s performance and reduces the dimensionality.
- **User Input**: The app has an interactive sidebar where users can select feature values via sliders. The features used for prediction are the most relevant ones selected during the feature selection phase.
- **Real-Time Prediction**: After the user selects feature values and clicks the "Predict" button, the app will display the predicted class (benign or malignant) and the prediction probability in real-time.

## Project Structure

The project is divided into several Python scripts, each focusing on specific parts of the pipeline:

- **`data_preparation.py`**: Handles data loading and preprocessing (including the use of SMOTE for class balancing).
- **`feature_selection.py`**: Handles feature selection using `SelectKBest` for selecting the most important features.
- **`grid_search.py`**: Contains grid search for hyperparameter tuning (optional for future enhancement).
- **`ann_model.py`**: Contains the architecture and training of the artificial neural network (ANN) model.
- **`app.py`**: The main Streamlit app that combines all parts together and serves the user interface.

## Project Structure

The project is divided into the following files:

### 1. **data_preparation.py**
   - **Purpose**: Handles dataset loading, splitting, and preprocessing, including addressing class imbalance using **SMOTE**.
   - **Functions**:
     - `load_data()`: Loads the breast cancer dataset and prepares it for further processing.
     - `apply_smote(X, y)`: Applies **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the classes in the dataset.

### 2. **feature_selection.py**
   - **Purpose**: Performs feature selection using **SelectKBest** and **f_classif** to identify the most relevant features for the model.
   - **Functions**:
     - `select_features(X, y)`: Selects the top k features that are most significant for the classification task.

### 3. **grid_search.py** (Optional for future enhancement)
   - **Purpose**: This file can be used to perform **hyperparameter tuning** using **GridSearchCV** to find the best parameters for the model.
   - **Note**: It is not yet implemented but can be added later to enhance model performance by fine-tuning hyperparameters like the number of hidden layers, activation functions, etc.

### 4. **ann_model.py**
   - **Purpose**: Contains the code to build and train the **Artificial Neural Network (ANN)** model for breast cancer classification.
   - **Functions**:
     - `train_ann_model(X_train, y_train)`: Trains the ANN model using the provided training data (`X_train` and `y_train`) and returns the trained model.

### 5. **app.py**
   - **Purpose**: The main **Streamlit** app that ties everything together. It allows users to input tumor feature values through the Streamlit sidebar and provides predictions based on those inputs.
   - **Features**:
     - Users can input tumor features via sliders.
     - The app scales the user input using the same scaler used in training.
     - The app makes predictions using the trained ANN model and displays the results, including the prediction (benign or malignant) and the prediction probabilities.

## How the App Works

1. **Data Loading and Preprocessing**:
   - The breast cancer dataset is loaded, and **SMOTE** is applied to balance the classes.
   
2. **Feature Selection**:
   - The most important features are selected using **SelectKBest** and **f_classif**.

3. **Model Training**:
   - An **Artificial Neural Network (ANN)** model is trained on the processed data.
   
4. **User Input**:
   - The user provides input through a Streamlit sidebar, adjusting sliders for tumor features (e.g., mean radius, area, concavity).
   
5. **Prediction**:
   - The user input is scaled using the same scaler used during model training.
   - The model makes a prediction based on the input and displays the result, including the class (benign or malignant) and the prediction probabilities.

## Requirements

To run the app, you will need the following Python libraries:

- `streamlit`
- `scikit-learn`
- `imblearn`
- `pandas`
- `numpy`

You can install these dependencies using `pip`:

```bash
pip install streamlit scikit-learn imbalanced-learn pandas numpy

## Installation

**To run this project locally, follow these steps:**

1. **Clone the repository to your local machine.**

   ```bash
   git clone https://github.com/tamannada26/breast-cancer-prediction-app.git
