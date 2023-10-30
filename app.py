#with streamlit

import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

sclr = StandardScaler()

# Load your trained RandomForestRegressor model
with open('model_rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def main():

    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1574375927938-d5a98e8ffe85?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1yZWxhdGVkfDE1fHx8ZW58MHx8fHx8&w=1000&q=80");
            background-size: 100% 100%;
            background-repeat: no-repeat;
            background-position: center center;
            width: 100%;
            height: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Stock Price Prediction')

    # Add a sidebar
    st.sidebar.title('Input Data')

    # User input form in the sidebar
    open_val = st.sidebar.number_input('Open Value')
    high_val = st.sidebar.number_input('High Value')
    low_val = st.sidebar.number_input('Low Value')
    adj_close_val = st.sidebar.number_input('Adjusted Close Value')
    volume_val = st.sidebar.number_input('Volume')
    year_val = st.sidebar.number_input('Year', min_value=2000, max_value=2023)
    month_val = st.sidebar.number_input('Month', min_value=1, max_value=12)
    day_val = st.sidebar.number_input('Day', min_value=1, max_value=31)

    if st.sidebar.button('Predict'):
        try:
            # Prepare the input data for prediction
            input_data = np.array([[open_val, high_val, low_val, adj_close_val, volume_val, year_val, month_val, day_val]])

            # Standardize the input data
            input_data = sclr.fit_transform(input_data)

            # Make a prediction using the model
            prediction = model.predict(input_data)

            st.success(f'Predicted Close: {prediction[0]:.2f}')
        except Exception as e:
            st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main()
