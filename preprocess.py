# preprocess.py

import pandas as pd
import os

# Load the dataset
df = pd.read_csv("data/student_marks.csv")

print("Original Dataset:")
print(df)

# Remove duplicate rows (if any)
df = df.drop_duplicates()

# Remove missing values (if any)
df = df.dropna()

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save the processed dataset
df.to_csv("data/processed.csv", index=False)

print("\nProcessed dataset saved successfully!")
print("Location: data/processed.csv")