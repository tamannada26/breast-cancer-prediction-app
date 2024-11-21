
# Breast Cancer Prediction App

This is a machine learning application built with Streamlit for predicting whether a tumor is **Malignant** (cancerous) or **Benign** (non-cancerous) based on the **Breast Cancer Dataset**. The app uses an Artificial Neural Network (ANN) model to make predictions and provides users with the probability of malignancy.

## Key Features of the App:

1. **User Input via Sidebar**:  
   - The app provides interactive sliders in the sidebar for users to input the values for different tumor features. These features include various metrics such as **mean radius**, **mean perimeter**, **mean area**, etc.
  
2. **Prediction and Probability**:  
   - After inputting the feature values, users can click the "Predict" button to receive the prediction. The app also shows the **prediction probability**, which indicates how confident the model is about its prediction.
   
3. **Visual Representation**:  
   - The app displays an optional image for **Breast Cancer Awareness** and the **dataset** when the user opts to view it.

4. **Feature Descriptions**:  
   - The app provides clear and concise descriptions of each feature used for prediction, helping users understand the meaning behind the inputs they provide.



## Installation Instructions

1. Clone the repository to your local machine:

    ```bash
   git clone https://github.com/tamannada26/breast-cancer-prediction-app.git

    ```

2. Navigate to the project folder:

    ```bash
    cd breast-cancer-prediction-app
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    This will install:
    - Streamlit
    - Pandas
    - Scikit-learn
    - Imbalanced-learn
    - NumPy

## How to Run the App

1. Ensure that the required dependencies are installed (follow the installation instructions above).
   
2. Run the app using Streamlit:

    ```bash
    streamlit run app.py
    ```

3. The app will open in your default web browser. If not, you can access it at `http://localhost:8501`.

## Files in the Project

- **`app.py`**: The main Streamlit application that integrates all modules and provides the user interface.
- **`data_preparation.py`**: Handles the dataset loading, splitting, and preprocessing (including SMOTE).
- **`feature_selection.py`**: Performs feature selection using SelectKBest and f_classif.
- **`ann_model.py`**: Contains the ANN model and the function to train the model.
- **`grid_search_cv.py`**: (Optional for future enhancement) This file can be used for performing hyperparameter tuning using GridSearchCV on the model.

## File Descriptions

### `data_preparation.py`
This file handles the dataset loading, preprocessing, and class imbalance handling. It contains the following functions:
- `load_data()`: Loads the breast cancer dataset from scikit-learn.
- `apply_smote(X, y)`: Applies **SMOTE** to balance the class distribution in the dataset.

### `feature_selection.py`
This file is responsible for feature selection using the **SelectKBest** method from scikit-learn. It contains:
- `select_features(X, y, k=10)`: Selects the top `k` features based on the **ANOVA F-value** (using `f_classif`), improving model performance by focusing on relevant features.

### `grid_search_cv.py` (Optional)
This file is intended for **hyperparameter tuning** using **GridSearchCV**. It allows you to perform a search over a range of hyperparameters for the ANN model, helping to find the best set of parameters. (Note: This file is currently optional and can be used for future improvements.)

### `ann_model.py`
This file contains the **Artificial Neural Network (ANN)** model. It includes the following function:
- `train_ann_model(X_train, y_train)`: Trains the neural network model using the training data and returns the trained model. The model uses the **ReLU** activation function and the **Adam** optimizer.

### `app.py`
This is the main **Streamlit** application file. It integrates all the previous files and provides the user interface to interact with the model. Key steps include:
- Loading the dataset.
- Preprocessing the data with SMOTE.
- Selecting the most important features.
- Training and using the ANN model to make predictions.
- Displaying prediction results and visualizations.
- Providing an interactive sidebar for users to input tumor features and get predictions.

The app allows users to easily input feature values, make predictions, and view the associated probabilities and explanations of features.


## Example of Output

After selecting the tumor features and clicking the **"Predict"** button, the app will display something like:
Prediction: Benign (Non-cancerous) Prediction Probability: [0.95, 0.05]

This indicates that the tumor is predicted to be **Benign** with a high probability of 95% and a 5% probability of being **Malignant**.

#

## Acknowledgements

- The dataset is from the **UCI Machine Learning Repository**.
- Libraries like **Streamlit**, **Scikit-learn**, and **Imbalanced-learn** are used to build and run the app.

## Requirements
  - Streamlit
    - Pandas
    - Scikit-learn
    - Imbalanced-learn
    - NumPy


