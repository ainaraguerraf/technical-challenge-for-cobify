import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('data_model.csv')

# Preprocess the data
data['gas_type'] = data['gas_type'].map({'E10': 0, 'SP98': 1})
selected_columns = ['distance', 'consume', 'speed', 'temp_inside', 'temp_outside', 'AC', 'rain', 'sun', 'snow']
X = data[selected_columns]
y = data['gas_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
st.title('🚗⛽✨ Check out which fuel is recommended for your next ride Cobyfier')

st.markdown(f'## How do you think your next ride will be? 🪄')

distance = st.number_input('Distance (km)', min_value=0.0, max_value=1000.0, step=0.1)
consume = st.number_input('Fuel consumption (L/100km)', min_value=0.0, max_value=30.0, step=0.1)
speed = st.number_input('Average speed (km/h)', min_value=0, max_value=200, step=1)
temp_inside = st.number_input('Temperature inside (Celsius)', min_value=-10, max_value=40, step=1)
temp_outside = st.number_input('Temperature outside (Celsius)', min_value=-10, max_value=40, step=1)
rain = st.checkbox('Rain ☔')
sun = st.checkbox('Sun ☀️')
snow = st.checkbox('Snow ❄️')
ac = st.checkbox('AC ❄️')

input_data = [[distance, consume, speed, temp_inside, temp_outside, int(rain), int(sun), int(snow), int(ac)]]

if st.button('Predict my fuel'):
    prediction = model.predict(input_data)
    if prediction[0] == 0:
        result = 'E10'
    else:
        result = 'SP98'
    st.markdown(f"## Based on your data, you should choose **{result}**. Have a nice ride! 🚀")

st.write(f"Accuracy of the model: {accuracy:.2f}")
