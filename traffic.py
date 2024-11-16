# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up the app title and image
st.title('Traffic Volume Predictor')
st.write("Utilize our Advanced Machine Learning Algorithm to Predict Traffic Volume!")
st.image('traffic_image.gif', use_column_width = True) 

# Reading the pickle file that we created before 
model_pickle = open('reg_traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('example_data.csv')

# Sidebar for user inputs with an expander
st.sidebar.image('traffic_sidebar.jpg', use_column_width = True, 
         caption = "Traffic Volume Predictor")
st.sidebar.title('Input Features')
st.sidebar.write("Upload your data file or maunally enter your specifications")

with st.sidebar.expander("Option 1: Upload CSV File"):
    user_csv = st.file_uploader('')
    st.title('Sample Data for Upload')
    st.dataframe(default_df.head())
    st.write("Make sure your file has the same columns and data types as shown above")

with st.sidebar.expander("Option 2: Fill Out Form"):
    st.header("Enter Your Details Below")
    with st.form('User Input Form'):
        holiday = st.selectbox('Choose whether today is a holiday or not', options=[None, 'Labor Day', 'Thanksgiving Day', 
                                                                                    'Christmas Day', 'New Years Day',
                                                                                    'Martin Luther King Jr Day', 'Columbus Day',
                                                                                    'Veterans Day', 'Washingtons Birthday',
                                                                                    'Memorial Day', 'Independence Day', 'State Fair'])
        temp = st.number_input('Average Temperature in Kelvin', min_value=0.0, max_value=500.0, step=0.1, help="Range: 0-500")
        rain_1h = st.number_input('Amount in mm of rain for the past hour', min_value=0.0, max_value=10000.0, step=0.1, help="Range: 0-10000")
        snow_1h = st.number_input('Amount in mm of snow for the past hour', min_value=0.0, max_value=1.0, step=0.01, help="Range: 0-1")
        clouds_all = st.number_input('Percentage of Cloud Cover', min_value=0, max_value=100, step=1, help="Range: 0-100")
        weather_main = st.selectbox('Choose the Current Weather', options=['Clouds', 'Clear', 'Mist', 'Rain', 'Snow', 'Drizzle', 
                                                                           'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall'])
        month = st.selectbox('Choose Month', options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                                                      'September', 'October', 'November', 'December'])
        weekday = st.selectbox('Choose Day of the Week', options=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
        hour = st.selectbox('Choose Hour', options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        submit_button = st.form_submit_button("Predict")

# Combine the list of user data as a row to default_df, turn hours back into 'object'
default_df.loc[len(default_df)] = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, month, weekday, hour]
default_df['hour'] = default_df['hour'].astype('object')

 # Create dummies for encode_df
encode_dummy_df = pd.get_dummies(default_df)

# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

# Get the prediction with its intervals
alpha = st.slider("Enter your alpha value", min_value=.01, max_value=.5, value = 0.1, step = 0.01)
prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
pred_value = prediction[0]
lower_limit = intervals[:, 0]
upper_limit = intervals[:, 1][0][0]

# Ensure lower limit is above 0
lower_limit = max(0, lower_limit[0][0])

# Show the prediction on the app
st.write("## Predicting Traffic Volume")

if user_csv is not None:
    #attaching user data on bottom of default dataset
    user_df = pd.read_csv(user_csv)
    user_df['hour'] = user_df['hour'].astype('object')
    user_df['holiday'] = user_df['holiday'].replace(to_replace=np.NaN, value=None)

    attached_df = pd.concat([default_df, user_df])

    #get dummies and extract just the user data
    big_encode_dummy_df = pd.get_dummies(attached_df)
    big_user_encoded_df = big_encode_dummy_df.tail(len(user_df))

    #make predictions on user data
    prediction, intervals = reg_model.predict(big_user_encoded_df, alpha = alpha)

    #append results to original dataframe
    user_df['Prediction'] = prediction.astype('int')
    user_df['Lower CI Limit'] = intervals[:, 0].round(0)
    user_df['Lower CI Limit'] = user_df['Lower CI Limit'].apply(lambda x: max(0, x))
    user_df['Upper CI Limit'] = intervals[:, 1].round(0)

    st.write("With a", ((1 - alpha) * 100), "% confidence interval:")
    st.write(user_df)
else:
    # Display results using metric card
    st.metric(label = "Predicted Volume", value = f"{int(pred_value)}")
    st.write("With a", ((1 - alpha) * 100), "% confidence interval:")
    st.write(f"**Confidence Interval**: [{lower_limit.round(0)}, {upper_limit.round(0)}]")

# Additional tabs for XGBoost model performance
st.subheader("Model Insights")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")
