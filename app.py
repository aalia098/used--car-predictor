"""
Small web app for the project.  Run:  streamlit run app.py
(train the models first with:  python train.py)
"""
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Checker", page_icon="🚗")

@st.cache_resource
def load_everything():
    price_model = joblib.load("models/price_model.joblib")
    deal_model = joblib.load("models/deal_model.joblib")
    with open("models/options.json") as f:
        options = json.load(f)
    return price_model, deal_model, options

try:
    price_model, deal_model, options = load_everything()
except FileNotFoundError:
    st.error("Models not found. Run `python train.py` first.")
    st.stop()

st.title("🚗 Used Car Price and Deal Checker")
st.write("Enter the car details and the price the seller is asking. "
         "I'll estimate what the car should cost and tell you if it's a good deal.")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", options["brands"])
    fuel = st.selectbox("Fuel type", options["fuels"])
    transmission = st.selectbox("Transmission", options["transmissions"])
with col2:
    age = st.slider("Age of car (years)", 0, options["max_age"], 5)
    km = st.number_input("Kilometres driven", 0, 400000, 50000, step=1000)
    owner = st.selectbox("Number of owners", [1, 2, 3])

asking = st.number_input("Seller's asking price (in lakh ₹)", 0.1, 100.0, 4.0, step=0.1)

if st.button("Check this deal"):
    car = {"brand": brand, "fuel": fuel, "transmission": transmission,
           "age": age, "km_driven": km, "owner": owner}

    fair = price_model.predict(pd.DataFrame([car]))[0]
    diff = (asking - fair) / fair * 100

    # the classifier works on the fair price, the asking price and the gap
    deal_input = pd.DataFrame([{"fair_price": fair, "asking_price": asking, "gap_pct": diff}])
    label = deal_model.predict(deal_input)[0]

    st.subheader(f"Estimated fair price: ₹ {fair:.2f} lakh")
    if label == "Good Deal":
        st.success(f"Good Deal: the asking price is about {abs(diff):.0f}% below the estimate.")
    elif label == "Overpriced":
        st.error(f"Overpriced: the asking price is about {abs(diff):.0f}% above the estimate.")
    else:
        st.info("Fair: the asking price is close to the estimate.")

    # confidence of the classifier for each label
    probs = deal_model.predict_proba(deal_input)[0]
    st.bar_chart(pd.Series(probs, index=deal_model.classes_))
