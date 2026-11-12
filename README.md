# Used Car Price and Deal Quality Predictor

A supervised learning project in Python.

> It predicts the resale price of a used car using **regression**, then tells you
> whether the seller's asking price is a **Good Deal, Fair, or Overpriced** using **classification**.

## How it works

```
car details ──► Regression model ──► fair price ─┐
                                                 ├─► Classification model ──► Good Deal / Fair / Overpriced
seller's asking price ───────────────────────────┘
```

| Part | Type | Question | Output |
|------|------|----------|--------|
| 1 | Regression | What should this car cost? | A price in lakh ₹ |
| 2 | Classification | Is the asking price a good deal? | A label |

The classifier takes three inputs: the fair price from part 1, the asking price, and the gap between them in %.

## Folder layout

```
used-car-predictor/
├── make_data.py        creates a practice dataset (data/cars.csv)
├── train.py            cleans data, makes graphs, trains and saves both models
├── app.py              Streamlit web app
├── requirements.txt
├── data/               cars.csv lives here
├── models/             saved models (created by train.py)
└── reports/            graphs (created by train.py)
```

## How to run

You need Python 3.9 or newer.

```bash
# 1. (optional but recommended) make a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 2. install the libraries
pip install -r requirements.txt

# 3. create the dataset (skip this if you are using the real Kaggle CSV, see below)
python make_data.py

# 4. train the models, prints the scores and saves graphs to reports/
python train.py

# 5. open the web app
streamlit run app.py
```

The app opens in your browser at http://localhost:8501.

## Using the real CarDekho dataset (recommended for submission)

`make_data.py` generates a synthetic dataset so the project runs anywhere.
For a stronger project, use real data:

1. Download a used-car CSV from Kaggle (search "CarDekho used car data").
2. Save it as `data/cars.csv`.
3. Make sure it has these columns (train.py already renames the common Kaggle names):
   `brand, year, km_driven, fuel, transmission, owner, selling_price`
4. If `selling_price` is in rupees instead of lakh, divide it by 100000 first.
5. Some Kaggle files store owner as text ("First Owner"). Convert it to 1, 2, 3.
6. Run `python train.py` and `streamlit run app.py` as before.

## Models and metrics

| Task | Baseline | Final model | Metrics |
|------|----------|-------------|---------|
| Regression | Linear Regression | Random Forest | MAE, R² |
| Classification | Logistic Regression | whichever scores higher | Accuracy, confusion matrix |

Results on the synthetic data (yours will change slightly with real data):

- Regression: Random Forest MAE ≈ 0.29 lakh, R² ≈ 0.98 (Linear Regression: MAE ≈ 1.06, R² ≈ 0.80)
- Classification: accuracy ≈ 76%

The classification score is lower than the regression score on purpose, and it is not a bug.
Cars near the 15% boundary are hard to label because the predicted price itself has a small error.

## Key design decisions (good for viva)

- **80/20 train-test split**, so the score is measured on cars the model has never seen.
- **One-hot encoding** for text columns (brand, fuel, transmission); **scaling** for numbers.
- **Year is converted to Age**, which is easier for a model to use.
- **Deal label rule:** asking price more than 15% above fair = Overpriced, more than 15% below = Good Deal, otherwise Fair.
- **Asking price is simulated.** Public datasets only contain the price a car sold for, not the original ask.
  So `train.py` creates an asking price by multiplying the fair price by a random factor (0.75 to 1.35).
  Be upfront about this in your viva. It is a limitation, and it is also a reasonable way to build labelled data.
- **cross_val_predict is used when training the classifier**, so the fair prices it learns from
  come from a model that has not seen those cars. Without this the classifier would see unrealistically
  perfect prices during training.

## Limitations and future scope

- No city, accident history, or car variant (VXi, ZXi, etc.) in the features.
- Asking prices are simulated, not scraped from real listings.
- Could be extended with live listings and deployed online.

## Graphs produced (in `reports/`)

1. Price vs age
2. Price vs kilometres driven
3. Price by fuel type
4. Actual vs predicted price
5. Confusion matrix
