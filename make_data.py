"""
Makes a practice dataset (data/cars.csv) so the project runs out of the box.

It is synthetic: prices follow simple real-world rules (older car = cheaper,
more km = cheaper, diesel/automatic = a bit costlier, more owners = cheaper)
plus some random noise. If you have the real CarDekho CSV from Kaggle,
put it in data/cars.csv instead and skip this script (see README).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000
CURRENT_YEAR = 2025

# rough new-car price in lakh rupees
BRANDS = {
    "Maruti": 6, "Hyundai": 7, "Tata": 7, "Honda": 10, "Mahindra": 12,
    "Volkswagen": 10, "Kia": 12, "Toyota": 15, "Ford": 8, "BMW": 45,
}

brand = rng.choice(list(BRANDS), size=N, p=[.24, .18, .11, .10, .09, .06, .06, .06, .05, .05])
year = rng.integers(2008, 2025, size=N)
age = CURRENT_YEAR - year

# people drive ~10-12k km a year, with a lot of variation
km = (age * rng.normal(11000, 3500, size=N)).clip(1500, None).round(-2)
fuel = rng.choice(["Petrol", "Diesel", "CNG"], size=N, p=[.55, .38, .07])
transmission = rng.choice(["Manual", "Automatic"], size=N, p=[.78, .22])
owner = rng.choice([1, 2, 3], size=N, p=[.65, .27, .08])

new_price = np.array([BRANDS[b] for b in brand], dtype=float)
price = new_price * np.exp(-0.11 * age)          # loses ~10% value every year
price *= np.exp(-km / 400000)                    # heavy use lowers it a bit more
price *= np.where(fuel == "Diesel", 1.08, np.where(fuel == "CNG", 0.95, 1.0))
price *= np.where(transmission == "Automatic", 1.12, 1.0)
price *= 1 - 0.06 * (owner - 1)
price *= rng.lognormal(0, 0.07, size=N)          # every car is a little different

df = pd.DataFrame({
    "brand": brand, "year": year, "km_driven": km, "fuel": fuel,
    "transmission": transmission, "owner": owner,
    "selling_price": price.round(2).clip(0.3, None),   # in lakh
})
df.to_csv("data/cars.csv", index=False)
print(f"Saved {len(df)} rows to data/cars.csv")
