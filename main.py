import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('data.csv')

# Split the dataset into features (X) and the target variable (y)
X = df.drop(columns=['label'])
y = df['label']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
dec_tree = DecisionTreeClassifier(criterion="entropy", splitter="best", max_features=7, max_depth=6)
dec_tree.fit(x_train, y_train)

# Streamlit web application
st.set_page_config(layout="centered")  # Center align the app content
st.title('Crop Label Prediction')
st.markdown("---")

# User input for feature values
st.sidebar.title('Input Values')

N = st.sidebar.slider('Nitrogen (N)', min_value=0, max_value=100, value=50, step=1)
P = st.sidebar.slider('Phosphorus (P)', min_value=0, max_value=100, value=50, step=1)
K = st.sidebar.slider('Potassium (K)', min_value=0, max_value=100, value=50, step=1)
temperature = st.sidebar.slider('Temperature', min_value=0, max_value=50, value=25, step=1)
humidity = st.sidebar.slider('Humidity', min_value=0, max_value=100, value=50, step=1)
ph = st.sidebar.slider('pH', min_value=0, max_value=140, value=70, step=1)  # Adjusted step and value for pH
rainfall = st.sidebar.slider('Rainfall', min_value=0, max_value=1000, value=500, step=10)

# Predict button
st.sidebar.markdown("---")
if st.sidebar.button('Predict', key="predict_button"):
    # Prepare user input as a DataFrame
    user_input = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'temperature': [temperature],
                               'humidity': [humidity], 'ph': [ph / 10], 'rainfall': [rainfall]})
    # Predict the label
    prediction = dec_tree.predict(user_input)
    
    # Display the prediction
    st.markdown("---")
    st.subheader('Prediction')
    st.write('Predicted Label:', prediction[0])
