# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preparation import load_data, apply_smote
from feature_selection import select_features
from ann_model import train_ann_model
from grid_search_cv import grid_search_ann

# Title of the Streamlit App
st.title("Breast Cancer Prediction App")

# Step 1: Load the dataset
df = load_data()

# Step 2: Data Preparation
X = df.drop(columns=['target'])
y = df['target']

# Apply SMOTE to handle class imbalance
X_resampled, y_resampled = apply_smote(X, y)

# Step 3: Feature Selection
X_selected, selected_features = select_features(X_resampled, y_resampled)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.3, random_state=42)

# Step 5: Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train ANN Model
model = train_ann_model(X_train_scaled, y_train)

# Step 7: Sidebar for User Input
st.sidebar.header("Select values for Tumor's features")
user_inputs = {}

# Add sliders dynamically for selected features
for feature in selected_features:
    user_inputs[feature] = st.sidebar.slider(feature, min_value=float(X[feature].min()), 
                                             max_value=float(X[feature].max()), 
                                             value=float(X[feature].mean()))

# Convert user inputs into a DataFrame
user_input_df = pd.DataFrame([user_inputs])

# Step 8: Scale User Input
user_input_scaled = scaler.transform(user_input_df[selected_features])

# Step 9: Predict and Display Results
if st.button('Predict'):
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)
    if prediction[0] == 0:
        st.write("Prediction: **Malignant** (Cancerous)")
    else:
        st.write("Prediction: **Benign** (Non-cancerous)")

    st.write(f"Prediction Probability: {prediction_proba[0]}")

# Step 10: Display Image (Optional)
st.image("breast_cancer_awareness.jpg", use_column_width=True)

# Step 11: Feature Description Section
st.markdown("## **Description of Features**:")
st.markdown("- **mean radius**: Average distance from the center to points on the tumor's perimeter.")
st.markdown("- **mean perimeter**: Average length of the tumor's boundary.")
st.markdown("- **mean area**: Average area of the tumor.")
st.markdown("- **mean concavity**: Severity of concave portions of the tumor's contour.")
st.markdown("- **mean concave points**: Number of concave portions on the tumor.")

# Add the explanation for the 'worst' features
st.markdown("- **Worst Radius**: The largest distance from the center to the perimeter of the tumor, indicating tumor size.")
st.markdown("- **Worst Area**: The largest area enclosed by the tumor's boundary, indicating tumor size.")
st.markdown("- **Worst Concavity**: The most concave (inward bulging) portion of the tumor's contour, often found in malignant tumors.")
st.markdown("- **Worst Concave Points**: The largest number of inward bulges or indentations on the tumor's boundary, indicative of malignancy.")
st.markdown("- **Worst Perimeter**: The largest length around the tumor’s boundary, reflecting tumor size and potential malignancy.")

# Step 12: Optional Visualization
if st.checkbox('Show Dataset'):
    st.dataframe(df)
