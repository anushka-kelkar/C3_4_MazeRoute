import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score

# Load dataset
df = pd.read_csv('terrain_datasetC1.csv')

# Identify feature columns (everything except 'terrain' and 'difficulty')
feature_cols = [col for col in df.columns if col not in ['terrain', 'difficulty']]
X = df[feature_cols]
y_class = df['terrain']
y_reg = df['difficulty']

# Encode terrain labels (for classification)
label_encoder = LabelEncoder()
y_class_encoded = label_encoder.fit_transform(y_class)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for training/testing
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X_scaled, y_class_encoded, test_size=0.2, random_state=42
)
_, _, y_reg_train, y_reg_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

# Train models
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_class_train)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_reg_train)

# Save models (optional)
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/terrain_classifier.pkl")
joblib.dump(reg, "models/difficulty_regressor.pkl")
joblib.dump(scaler, "models/feature_scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

# Functions for prediction
def predict_terrain_type(features):
    """
    Predicts the type of terrain (classification).
    Input: features (list or numpy array of feature values)
    Output: predicted terrain label
    """
    features_scaled = scaler.transform([features])
    pred_encoded = clf.predict(features_scaled)[0]
    terrain_label = label_encoder.inverse_transform([pred_encoded])[0]
    return terrain_label

def predict_difficulty(features):
    """
    Predicts the numeric difficulty value (regression).
    Input: features (list or numpy array of feature values)
    Output: predicted difficulty (float)
    """
    features_scaled = scaler.transform([features])
    predicted_diff = reg.predict(features_scaled)[0]
    return float(predicted_diff)

def get_user_input():
    """
    Get terrain features from user input.
    Returns: list of feature values
    """
    print("\n" + "="*60)
    print("TERRAIN PREDICTION - Feature Input")
    print("="*60)
    print("\nPlease enter the following terrain features:")
    print("-"*60)
    
    features = []
    for i, col in enumerate(feature_cols, 1):
        while True:
            try:
                # Display feature name in a readable format
                feature_name = col.replace('_', ' ').title()
                value = input(f"{i}. {feature_name}: ")
                
                # Try to convert to float
                feature_value = float(value)
                features.append(feature_value)
                break
            except ValueError:
                print(f"  Invalid input! Please enter a numeric value.")
    
    return features

def display_prediction_results(terrain_type, difficulty):
    """
    Display prediction results in a formatted way.
    """
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Terrain Type: {terrain_type}")
    print(f"Predicted Difficulty:   {difficulty:.2f}")
    print("="*60)

def get_feature_statistics():
    """
    Display statistics about the features to help users.
    """
    print("\n" + "="*60)
    print("FEATURE STATISTICS (for reference)")
    print("="*60)
    stats = df[feature_cols].describe()
    print(stats.round(2))
    print("="*60)

# Main execution block - only runs when script is executed directly
if __name__ == "__main__":
    print("\n🚀 Terrain Prediction Model")
    print("This model predicts terrain type and difficulty based on input features.\n")
    
    while True:
        print("\nOptions:")
        print("1. Enter features manually")
        print("2. View feature statistics")
        print("3. Use sample data")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            # Manual input
            features = get_user_input()
            terrain_type = predict_terrain_type(features)
            difficulty = predict_difficulty(features)
            display_prediction_results(terrain_type, difficulty)
            
        elif choice == "2":
            # Show statistics
            get_feature_statistics()
            
        elif choice == "3":
            # Use sample data
            sample_features = X.iloc[0].tolist()
            print("\n📝 Using sample features from dataset:")
            for col, val in zip(feature_cols, sample_features):
                print(f"   {col}: {val:.2f}")
            
            terrain_type = predict_terrain_type(sample_features)
            difficulty = predict_difficulty(features)
            display_prediction_results(terrain_type, difficulty)
            
        elif choice == "4":
            print("\n Exiting. Goodbye!")
            break
            
        else:
            print("\n Invalid option. Please select 1-4.")
        
        # Ask if user wants to continue
        continue_input = input("\nMake another prediction? (y/n): ").strip().lower()
        if continue_input != 'y':
            print("\n Exiting. Goodbye!")
            break