import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# 1. Load your cleaned data
df = pd.read_csv("cleaned_dataset.csv")

# 2. Select data for prediction
# Assuming the last column is the target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. Split data into "study" (training) and "exam" (testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train your model (the prediction machine)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 5. Test your model
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print("R² Score (Model Intelligence):", score)

# 6. Save your trained model for future use
joblib.dump(model, "trained_model.joblib")
joblib.dump(X.columns.tolist(), "model_columns.joblib")
print("\n✅ Model trained successfully and saved as 'trained_model.joblib'.")
