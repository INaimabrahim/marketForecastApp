import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the pre-trained model from pickle file
def load_gru_model():
    try:
        with open('best_model.pkl', 'rb') as file:
            gru_model = pickle.load(file)  # Load the model from pickle
        return gru_model
    except:
        st.error("Model file not found. Please train the model first.")
        return None

# Evaluate the model performance
def evaluate_model(true, predicted):
    mse = mean_squared_error(true, predicted)
    mae = mean_absolute_error(true, predicted)
    r2 = r2_score(true, predicted)
    return mse, mae, r2

# Preprocess the data (Scaling, etc.)
def preprocess_data(df, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['y']].values)
    
    # Creating sequences for training
    X, y = [], []
    for i in range(seq_length, len(df)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for GRU model input (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Streamlit app
st.title('CPI Prediction with GRU Model')

# Upload the test data (Replace with your actual data)
uploaded_file = st.file_uploader("Upload a Excel file for prediction", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['ds'] = pd.to_datetime(df['ds'])  # Assuming 'ds' is the datetime column

    # Show the raw data in the Streamlit app
    st.write("Raw Time Series Data", df.tail())

    # Load the pre-trained GRU model
    gru_model = load_gru_model()

    if gru_model is None:
        st.error("Model not found. Please train the model first.")
    else:
        # Preprocess the data
        seq_length = st.slider("Select sequence length", 10, 50, 20)
        X_test, y_test, scaler = preprocess_data(df, seq_length)

        # Predict with the GRU model
        gru_predictions = gru_model.predict(X_test)
        gru_predictions = scaler.inverse_transform(gru_predictions)

        # Show the predicted vs actual plot
        st.subheader("Predicted vs Actual Plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['ds'][seq_length:], df['y'][seq_length:], label='Actual', color='black')
        ax.plot(df['ds'][seq_length:], gru_predictions, label='Predicted', color='blue')
        ax.set_title("GRU Time Series Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

        # Evaluate the model performance
        mse, mae, r2 = evaluate_model(df['y'][seq_length:], gru_predictions)

        # Display the evaluation metrics
        st.subheader("Evaluation Metrics")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R-squared (R2): {r2:.2f}")
