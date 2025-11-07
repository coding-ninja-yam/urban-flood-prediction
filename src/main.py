from data_cleaning import clean_data
from feature_engineering import add_features
from model_training import train_model

if __name__ == "__main__":
    print("🌍 Starting Urban Flood Prediction Pipeline...")

    # Step 1: Clean
    df = clean_data("data/climate_data.csv")
    print("✅ Data cleaned.")

    # Step 2: Add Features
    df = add_features(df)
    print("✅ Feature engineering complete.")

    # Step 3: Train Model
    model, threshold, X_test, y_test, y_scores = train_model(df)
    print("✅ Model training complete.")
    print(f"🚀 Operational flood threshold = {threshold:.2f}")

