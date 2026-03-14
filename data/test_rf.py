import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

print("Loading data...")
df = pd.read_csv("clean_football_data.csv")

# 1. Prepare X (Features) and y (Target)
# We drop the Team Names because the math engine only understands numbers.
# We are forcing the model to predict purely based on the form/stats!
X = df.drop(columns=['HomeTeam', 'AwayTeam', 'Target'])
y = df['Target']

# 2. Split Data (Train on the first 80% of the season, test on the last 20%)
# shuffle=False is CRITICAL here. We can't train on future games to predict past games!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training on {len(X_train)} matches, testing on {len(X_test)} matches...")

# 3. Build and Train the Random Forest
# n_estimators=100 means we are building 100 different decision trees and making them vote
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# 4. Make Predictions on the Test Set
predictions = rf.predict(X_test)
acc = accuracy_score(y_test, predictions)



print("\n=== RESULTS ===")
print(f"Baseline Accuracy: {acc * 100:.2f}%")
print("\nDetailed Report (1=Home Win, 0=Draw, -1=Away Win):")
print(classification_report(y_test, predictions, zero_division=0))

# 5. Feature Importance (Why did it guess what it guessed?)
print("\nFeature Importance (What the model cares about most):")
for name, importance in zip(X.columns, rf.feature_importances_):
    print(f"{name}: {importance * 100:.1f}%")
