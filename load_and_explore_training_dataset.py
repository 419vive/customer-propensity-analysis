from pathlib import Path
import pandas as pd
import numpy as np

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

DATA_DIR = Path("/Users/jerrylaivivemachi/DS PROJECT/J_DA_Project/Customer propensity to purchase dataset")
TRAIN_CSV = DATA_DIR / "training_sample.csv"

df = pd.read_csv(TRAIN_CSV)

print(f"File: {TRAIN_CSV}")
print(f"Shape (rows, cols): {df.shape}\n")

print("Info:")
df.info(memory_usage="deep")
print()

print("Dtypes:")
print(df.dtypes.sort_index())
print()

print("Head:")
print(df.head(10).to_string(index=False))
print()

print("Missing values (top 20):")
missing = df.isna().sum().sort_values(ascending=False)
print(missing[missing > 0].head(20))
print()

numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print("Numeric summary:")
    print(df[numeric_cols].describe().T)
    print()

cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
if len(cat_cols) > 0:
    print("Categorical top values (up to 5 per column):")
    for col in sorted(cat_cols):
        vc = df[col].value_counts(dropna=False).head(5)
        print(f"\n{col}:")
        print(vc)

if "ordered" in df.columns:
    print("\nTarget 'ordered' distribution:")
    print(df["ordered"].value_counts(dropna=False))
    print("Proportion:")
    print((df["ordered"].value_counts(normalize=True, dropna=False) * 100).round(2).astype(str) + "%")


