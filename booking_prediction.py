import streamlit as st
import joblib
import numpy as np
import pandas as pd

try:
    model = joblib.load('XGB_booking.pkl')
    oneHot_encode_type = joblib.load('oneHot_encode_type.pkl')
    oneHot_encode_room = joblib.load('oneHot_encode_room.pkl')
    oneHot_encode_market = joblib.load('oneHot_encode_market.pkl')
    book_encode = joblib.load('booking_encode.pkl')
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")

def main():
    st.title('Hotel Booking Cancellation Prediction')

    # Input features one by one
    adults = st.number_input("Number of Adults", 0, 4)
    children = st.number_input("Number of Children", 0, 10)
    weekend_nights = st.number_input("Number of Weekend Nights", 0, 7)
    week_nights = st.number_input("Number of Week Nights", 0, 17)
    type_of_meal_plan = st.radio("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    car_parking = st.radio("Required Car Parking Space", [0, 1])
    room_type_reserved = st.radio("Room Type Reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    lead_time = st.number_input("Lead Time (days before arrival)", 0, 443)
    arrival_year = st.number_input("Arrival Year", 2017, 2018)
    arrival_month = st.number_input("Arrival Month", 1, 12)
    arrival_date = st.number_input("Arrival Date", 1, 31)
    market_segment_type = st.radio("Market Segment", ["Online", "Offline", "Corporate", "Complementary", "Aviation"])
    repeated_guest = st.radio("Is Repeated Guest", [0, 1])
    previous_cancellation = st.number_input("Previous Cancellations", 0, 13)
    previous_bookings = st.number_input("Previous Bookings Not Canceled", 0, 58)
    avg_price = st.number_input("Average Price Per Room (EUR)", 0.0, 540.0)
    special_requests = st.number_input("Number of Special Requests", 0, 5)

    # Combine into dictionary
    data = {
        'Number of Adults': int(adults), 'Number of Children': int(children),
        'Number of Weekend Nights': int(weekend_nights), 'Number of Week Nights': int(week_nights),
        'type_of_meal_plan': type_of_meal_plan, 'Car Parking': float(car_parking), 'room_type_reserved': room_type_reserved,
        'Lead Time': int(lead_time), 'Arrival Year': int(arrival_year), 'Arrival Month': int(arrival_month),
        'Arrival Date': int(arrival_date), 'market_segment_type': market_segment_type, 'Repeated Guest': int(repeated_guest),
        'Previous Cancellations': int(previous_cancellation), 'Previous Bookings Not Canceled': int(previous_bookings),
        'Avg Price Per Room': float(avg_price), 'Special Requests': int(special_requests)
    }

    df = pd.DataFrame([data])

    df['type_of_meal_plan'] = df['type_of_meal_plan'].astype(str)
    df['room_type_reserved'] = df['room_type_reserved'].astype(str)
    df['market_segment_type'] = df['market_segment_type'].astype(str)

    enc_meal = pd.DataFrame(
        oneHot_encode_type.transform(df[['type_of_meal_plan']]).toarray(),
        columns=oneHot_encode_type.get_feature_names_out(['type_of_meal_plan'])
    )

    enc_room = pd.DataFrame(
        oneHot_encode_room.transform(df[['room_type_reserved']]).toarray(),
        columns=oneHot_encode_room.get_feature_names_out(['room_type_reserved'])
    )

    enc_market = pd.DataFrame(
        oneHot_encode_market.transform(df[['market_segment_type']]).toarray(),
        columns=oneHot_encode_market.get_feature_names_out(['market_segment_type'])
    )

    # Gabung dan drop kolom aslinya
    df = pd.concat([df.reset_index(drop=True), enc_meal, enc_room, enc_market], axis=1)
    df = df.drop(['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], axis=1)


    if st.button('Make Prediction'):
        result = make_prediction(df)
        st.success(f'The prediction is: {"Canceled" if result == 0 else "Not Canceled"}')

def make_prediction(features):
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    main()