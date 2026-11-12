"""
Trains both models and saves them.

  Regression      -> predicts the fair price of a car (in lakh rupees)
  Classification  -> compares the asking price with that fair price and
                     says Good Deal / Fair / Overpriced

Run:  python train.py
"""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, mean_absolute_error, r2_score)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_YEAR = 2025
SEED = 42

# ---------------------------------------------------------------- 1. load
df = pd.read_csv("data/cars.csv")

# the Kaggle CarDekho files name their columns differently, so rename them
df = df.rename(columns={
    "Car_Name": "brand", "name": "brand", "Year": "year",
    "Selling_Price": "selling_price", "Kms_Driven": "km_driven",
    "Fuel_Type": "fuel", "fuel_type": "fuel", "Transmission": "transmission",
    "Owner": "owner",
})

# if the brand column holds full names like "Maruti Swift VDI", keep the first word
df["brand"] = df["brand"].astype(str).str.split().str[0]

# ---------------------------------------------------------------- 2. clean
needed = ["brand", "year", "km_driven", "fuel", "transmission", "owner", "selling_price"]
df = df[needed].dropna().drop_duplicates()
df["age"] = CURRENT_YEAR - df["year"]
df = df[df["age"] >= 0].drop(columns="year")
print(f"Rows after cleaning: {len(df)}")

# ------------------------------------------------- 3. make the deal label
# A real dataset only has the price a car sold for, not what the seller first
# asked. So we imitate sellers: the asking price is the fair price times a
# random factor between 0.75 and 1.35. Then we label it:
#   asking > 15% above fair  -> Overpriced
#   asking > 15% below fair  -> Good Deal
#   otherwise                -> Fair
rng = np.random.default_rng(SEED)
df["asking_price"] = (df["selling_price"] * rng.uniform(0.75, 1.35, len(df))).round(2)
ratio = df["asking_price"] / df["selling_price"]
df["deal"] = np.select([ratio > 1.15, ratio < 0.85], ["Overpriced", "Good Deal"], default="Fair")

# ---------------------------------------------------------------- 4. EDA
def save_plot(name):
    plt.tight_layout()
    plt.savefig(f"reports/{name}.png", dpi=120)
    plt.close()

plt.scatter(df["age"], df["selling_price"], s=6, alpha=0.4)
plt.xlabel("Age of car (years)"); plt.ylabel("Price (lakh)"); plt.title("Price vs age")
save_plot("1_price_vs_age")

plt.scatter(df["km_driven"], df["selling_price"], s=6, alpha=0.4, color="tab:green")
plt.xlabel("Kilometres driven"); plt.ylabel("Price (lakh)"); plt.title("Price vs kilometres")
save_plot("2_price_vs_km")

df.boxplot(column="selling_price", by="fuel", showfliers=False)
plt.suptitle(""); plt.title("Price by fuel type"); plt.ylabel("Price (lakh)")
save_plot("3_price_by_fuel")

# ------------------------------------------- 5. regression: fair price
cat_cols = ["brand", "fuel", "transmission"]
num_cols = ["age", "km_driven", "owner"]

def make_preprocessor(extra_num=()):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols + list(extra_num)),
    ])

X = df[cat_cols + num_cols]
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
}
print("\n=== REGRESSION (predicting price) ===")
best_reg, best_r2 = None, -1
for name, model in reg_models.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    r2 = r2_score(y_test, pred)
    print(f"{name:18s} MAE = {mean_absolute_error(y_test, pred):.2f} lakh   R2 = {r2:.3f}")
    if r2 > best_r2:
        best_reg, best_r2, best_pred = pipe, r2, pred

plt.scatter(y_test, best_pred, s=6, alpha=0.4)
lim = [0, y_test.max()]
plt.plot(lim, lim, color="red", label="perfect prediction")
plt.xlabel("Actual price (lakh)"); plt.ylabel("Predicted price (lakh)")
plt.title("Actual vs predicted price"); plt.legend()
save_plot("4_actual_vs_predicted")

# ------------------------------------- 6. classification: deal quality
# This is where the two parts connect. The classifier does NOT look at the
# car details directly. It looks at the fair price predicted by the regression
# model, the asking price, and how far apart they are (gap in %).
#
# For the training rows we use cross_val_predict, so each fair price comes from
# a model that never saw that car. Otherwise the predictions would look
# unrealistically perfect on training data and the classifier would be misled.
def deal_features(fair, asking):
    return pd.DataFrame({
        "fair_price": fair,
        "asking_price": asking,
        "gap_pct": (asking - fair) / fair * 100,
    })

train_fair = cross_val_predict(best_reg, X_train, y_train, cv=5)
test_fair = best_reg.predict(X_test)

Xc_train = deal_features(train_fair, df.loc[X_train.index, "asking_price"].values)
Xc_test = deal_features(test_fair, df.loc[X_test.index, "asking_price"].values)
yc_train = df.loc[X_train.index, "deal"]
yc_test = df.loc[X_test.index, "deal"]

clf_models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
}
print("\n=== CLASSIFICATION (deal quality) ===")
best_clf, best_acc = None, -1
for name, model in clf_models.items():
    model.fit(Xc_train, yc_train)
    pred = model.predict(Xc_test)
    acc = accuracy_score(yc_test, pred)
    print(f"{name:20s} accuracy = {acc:.3f}")
    if acc > best_acc:
        best_clf, best_acc, best_cpred, best_name = model, acc, pred, name

print(f"\nBest classifier: {best_name}")
print(classification_report(yc_test, best_cpred))
ConfusionMatrixDisplay.from_predictions(yc_test, best_cpred, cmap="Blues")
plt.title("Confusion matrix")
save_plot("5_confusion_matrix")

# ---------------------------------------------------------------- 7. save
joblib.dump(best_reg, "models/price_model.joblib")
joblib.dump(best_clf, "models/deal_model.joblib")

options = {
    "brands": sorted(df["brand"].unique().tolist()),
    "fuels": sorted(df["fuel"].unique().tolist()),
    "transmissions": sorted(df["transmission"].unique().tolist()),
    "max_age": int(df["age"].max()),
}
with open("models/options.json", "w") as f:
    json.dump(options, f)

print("\nSaved models to models/ and graphs to reports/")
