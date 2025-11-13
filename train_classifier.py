import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os
import sys

# --- 1. Load Data ---
data_file = 'augmented_medical_data_100k.csv'
if not os.path.exists(data_file):
    print(f"FATAL ERROR: Data file not found at '{data_file}'")
    print("Please make sure it's in the same folder as this script.")
    sys.exit(1)
    
print(f"Loading data from {data_file}...")
df = pd.read_csv(data_file)

# --- 2. Define Features (X) and Target (y) ---
# The target (y) is what we want to predict
y = df['breakdown_flag']

# The features (X) are the inputs used for prediction
# We drop the target column
X = df.drop('breakdown_flag', axis=1)

# --- 3. Preprocessing (CRITICAL STEP) ---
# ML models only understand numbers. We must convert 'device_name' (text)
# into numerical columns using OneHotEncoder.

categorical_features = ['device_name']
numeric_features = ['usage_hours', 'temperature', 'error_count']

# Create a "preprocessor" that applies the right transform to the right column
# 'passthrough' means the numeric features will be left alone
# 'OneHotEncoder' will be applied to the categorical 'device_name'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

print("Preprocessing data (One-Hot Encoding 'device_name')...")
# Fit the preprocessor to our data and transform it
X_processed = preprocessor.fit_transform(X)

# --- 4. Split Data ---
# Split into 80% for training and 20% for testing
print("Splitting data into 80% train and 20% test sets...")

# 'stratify=y' is VERY important. It ensures your 5% breakdown rate
# is correctly represented in both the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# --- 5. Train the New Classifier Model ---
print("Training the RandomForestClassifier...")
print("(This may take a minute with 100k rows...)")

# n_jobs=-1 uses all your computer's cores to train faster
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
model.fit(X_train, y_train)

print("Training complete.")

# --- 6. Evaluate the Model ---
# Let's see how well it learned
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Breakdown (1)']))

# --- 7. Save the New Model ---
# We must save BOTH the model AND the preprocessor,
# otherwise we can't transform new data in the app.
new_model_filename = 'trained_breakdown_classifier.pkl'

pipeline_to_save = {
    'preprocessor': preprocessor,
    'model': model
}

joblib.dump(pipeline_to_save, new_model_filename)
print(f"\n--- Success! ---")
print(f"New model and preprocessor saved to '{new_model_filename}'")