# Manishika Balamurugan
# Michigan Science Data Team
# Flight Price Prediction Project Team
# Presented at Michigan Institute for Data and AI in Society (MIDAS) Data Science Night

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import streamlit as st

# Random Forest Regressor Model
data = pd.read_csv("domestic.csv")

df_selected = data[["city1", "city2", "passengers", "fare_lg", "quarter", "carrier_lg"]]
df_selected = df_selected.dropna()

le_city1 = LabelEncoder()
le_city2 = LabelEncoder()
le_quarter = LabelEncoder()
le_carrier = LabelEncoder()

df_selected["city1_encoded"] = le_city1.fit_transform(df_selected["city1"])
df_selected["city2_encoded"] = le_city2.fit_transform(df_selected["city2"])
df_selected["quarter_encoded"] = le_quarter.fit_transform(df_selected["quarter"])
df_selected["carrier_encoded"] = le_carrier.fit_transform(df_selected["carrier_lg"])

X = df_selected[["city1_encoded", "city2_encoded", "quarter_encoded", "carrier_encoded"]]
y = df_selected["fare_lg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

with open("simple_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump((le_city1, le_city2, le_quarter, le_carrier), f)

# Streamlit App
st.title("Flight Price Predictor")

with open("simple_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    le_city1, le_city2, le_quarter, le_carrier = pickle.load(f)

city1_options = le_city1.classes_.tolist()
city2_options = le_city2.classes_.tolist()
quarter_options = le_quarter.classes_.tolist()
carrier_options = le_carrier.classes_.tolist()

quarter_display_map = {
    1: 'January - March',
    2: 'April - June',
    3: 'July - September',
    4: 'October - December'
}

full_names = {
    '9N': 'Tropic Air', 'A7': 'Unknown', 'AA': 'American Airlines',
    'AS': 'Alaska Airlines', 'B6': 'JetBlue Airways', 'CO': 'Continental Airlines',
    'DH': 'Independence Air', 'DL': 'Delta Air Lines', 'F9': 'Frontier Airlines',
    'FL': 'AirTran Airways', 'G4': 'Allegiant Air', 'HP': 'America West Airlines',
    'J7': 'ValuJet Airlines', 'JI': 'Midway Airlines', 'KP': 'Kiwi International Air Lines',
    'KW': 'Carnival Air Lines', 'MX': 'Mexicana', 'N7': 'National Airlines',
    'NJ': 'VivaAerobus', 'NK': 'Spirit Airlines', 'NW': 'Northwest Airlines',
    'PN': 'Pan Am', 'QQ': 'Reno Air', 'QX': 'Horizon Air',
    'RP': 'Chautauqua Airlines', 'RU': 'Unknown', 'SX': 'Skybus Airlines',
    'SY': 'Sun Country Airlines', 'T3': 'Eastern Air Lines', 'TW': 'TWA',
    'TZ': 'ATA Airlines', 'U5': 'USA3000 Airlines', 'UA': 'United Airlines',
    'US': 'US Airways', 'VX': 'Virgin America', 'W7': 'Western Pacific Airlines',
    'W9': 'Wizz Air', 'WN': 'Southwest Airlines', 'WV': 'Air Wisconsin',
    'XJ': 'Mesaba Airlines', 'XP': 'Casino Express', 'YV': 'Mesa Airlines',
    'YX': 'Republic Airlines', 'ZA': 'AccessAir', 'ZW': 'Air Wisconsin'
}

reverse_names = {display: original for original, display in full_names.items()}
display_to_original_quarter = {display: value for value, display in quarter_display_map.items()}
carrier_options_full = [full_names.get(code) for code in le_carrier.classes_]
quarter_options = [quarter_display_map[q] for q in le_quarter.classes_]

with st.form("input_form"):
    st.subheader("Enter Route Information")
    city1 = st.selectbox("Origin City", city1_options)
    city2 = st.selectbox("Destination City", city2_options)
    quarter = st.selectbox("Travel Quarter", quarter_options)
    carrier = st.selectbox("Airline Carrier", carrier_options_full)

    submitted = st.form_submit_button("Predict")

    if submitted:
        if city1 == city2:
            st.error("Origin and destination cities must be different.")
        else:

            city1_encoded = le_city1.transform([city1])[0]
            city2_encoded = le_city2.transform([city2])[0]
                
            quarter_original = display_to_original_quarter[quarter]
            quarter_encoded = le_quarter.transform([quarter_original])[0]
                
            carrier_code = reverse_names.get(carrier)
            carrier_encoded = le_carrier.transform([carrier_code])[0]

            input_data = np.array([[city1_encoded, city2_encoded, quarter_encoded, carrier_encoded]])
            predicted_fare = model.predict(input_data)[0]
            st.success(f"Estimated Fare: ${predicted_fare:.2f}")

# UI

with st.sidebar:
    st.header("📊 About This App")
    st.markdown("""
    This app predicts **one-way flight fares** using:

    - **Origin City**  
    - **Destination City**  
    - **Travel Quarter**  
    - **Airline Carrier**

    The model has an **R² score of 0.70** -- pretty solid for predicting your next ticket!

    ---

    ***Created by Manishika Balamurugan***  
    *Michigan Data Science Team*
    """)