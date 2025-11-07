import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

def train_model(df):
    # Label creation
    df['flood'] = (df['RR'] > 50).astype(int)

    features = [
        'Tn','Tx','Tavg','RH_avg','ss','ff_x','ddd_x','ff_avg',
        'month','rain_past3','rain_past7','RH_trend3','Tavg_trend3','rain_intensity'
    ]
    X = df[features]
    y = df['flood']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # XGBoost model
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=15,
        random_state=42
    )
    model.fit(X_res, y_res)

    # Evaluate threshold
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    best_threshold = thresholds[np.argmax(precision * recall)]

    print("🎯 Best F1 Threshold:", round(best_threshold, 2))
    print(classification_report(y_test, (y_scores >= best_threshold).astype(int)))

    # Save model
    joblib.dump(model, "models/xgb_flood_model.pkl")

    # Save PR curve
    plt.plot(recall, precision)
    plt.scatter(recall[np.argmax(precision * recall)],
                precision[np.argmax(precision * recall)],
                color='red', label='Best F1 point')
    plt.title("Precision–Recall Trade-off for Flood Detection")
    plt.xlabel("Recall (Floods Detected)")
    plt.ylabel("Precision (Correct Flood Alerts)")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/pr_curve.png")

    return model, best_threshold, X_test, y_test, y_scores

