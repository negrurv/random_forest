import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import time

# 1. Point Python to your compiled C++ library in the build folder
sys.path.append(os.path.join(os.path.dirname(__file__), "backend", "build"))

try:
    import rf_cpp
except ImportError:
    print("Could not find rf_cpp.so! Make sure you ran 'make' inside backend/build/")
    sys.exit(1)

print("Loading football data...")
df = pd.read_csv("data/clean_football_data.csv")

# ==========================================
# ⚠️ CHANGE THIS TO YOUR ACTUAL TARGET COLUMN NAME
TARGET_COL = 'Target' 
# ==========================================

if TARGET_COL not in df.columns:
    print(f"\nERROR: Please update TARGET_COL to match your CSV.")
    print(f"Here are your available columns:\n{list(df.columns)}")
    sys.exit(1)

# 2. Keep only numeric data (C++ can't do math on strings like "Arsenal")
# Also drop any rows with missing values (NaN) so they don't break the C++ doubles
df_numeric = df.select_dtypes(include=['number']).dropna()

X = df_numeric.drop(columns=[TARGET_COL])
y = df_numeric[TARGET_COL]

# 3. Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Flatten the 2D pandas DataFrames into 1D Python lists for our C++ bindings
X_train_flat = X_train.values.flatten().tolist()
y_train_list = y_train.values.tolist()

X_test_flat = X_test.values.flatten().tolist()
y_test_list = y_test.values.tolist()

num_samples_train, num_features = X_train.shape
num_samples_test, _ = X_test.shape

print(f"Training on {num_samples_train} matches with {num_features} stats per match...")

# 5. Initialize your custom C++ Random Forest
# Feel free to play with these hyperparameters!
rf = rf_cpp.RandomForest(
    num_trees=100, 
    max_depth=10, 
    min_samples_split=5, 
    feature_fraction=0.8
)

# 6. Train the model
start_time = time.time()
rf.train(X_train_flat, y_train_list, num_samples_train, num_features)
print(f"Training completed in {time.time() - start_time:.4f} seconds!")

# 7. Test the model's accuracy on unseen matches
print("Predicting unseen test matches...")
predictions = rf.predict_batch(X_test_flat, num_samples_test, num_features)

# Calculate Mean Squared Error (MSE)
mse = sum((p - a)**2 for p, a in zip(predictions, y_test_list)) / num_samples_test

print("-" * 40)
print(f"Test Mean Squared Error: {mse:.4f}")
print("If the MSE is low, your C++ AI is successfully predicting football!")
print("-" * 40)
