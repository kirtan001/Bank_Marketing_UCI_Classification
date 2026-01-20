import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.under_sampling import RandomUnderSampler


import os
import mlflow

os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("Bank_Marketing_Balanced_vs_Unbalanced")



# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("cleaned_bank_data.csv")
X = pd.get_dummies(df.drop(columns=["y"]), drop_first=True)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# -------------------------------
# Create balanced datasets
# -------------------------------
rus = RandomUnderSampler(random_state=42)
X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
X_test_bal, y_test_bal = rus.fit_resample(X_test, y_test)


# -------------------------------
# Evaluation function
# -------------------------------
def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob)
    }


# -------------------------------
# Model configurations
# -------------------------------
model_configs = {
    "DecisionTree": DecisionTreeClassifier(
        max_depth=6, min_samples_split=50, random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=50,
        n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )
}


results = {}
models = {}
run_ids = {}


# -------------------------------
# Training loop (Balanced vs Unbalanced)
# -------------------------------
for data_type, (X_tr, y_tr, X_te, y_te) in {
    "Unbalanced": (X_train, y_train, X_test, y_test),
    "Balanced": (X_train_bal, y_train_bal, X_test_bal, y_test_bal)
}.items():

    for model_name, model in model_configs.items():

        run_name = f"{model_name}_{data_type}"

        with mlflow.start_run(run_name=run_name) as run:
            model.fit(X_tr, y_tr)
            metrics = evaluate(model, X_te, y_te)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            key = f"{model_name}_{data_type}"
            results[key] = metrics
            models[key] = model
            run_ids[key] = run.info.run_id


# -------------------------------
# Compare all models
# -------------------------------
results_df = pd.DataFrame(results).T
results_df.to_csv("model_comparison.csv")

best_model_key = results_df["roc_auc"].idxmax()
best_model = models[best_model_key]
best_run_id = run_ids[best_model_key]

print(f"Best Overall Model: {best_model_key}")


# -------------------------------
# Save artifacts
# -------------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(best_model, "artifacts/best_model.pkl")
joblib.dump(X.columns.tolist(), "artifacts/feature_columns.pkl")


# -------------------------------
# Register best model
# -------------------------------
mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="BankMarketingBestModel"
)
