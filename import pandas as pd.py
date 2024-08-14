import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('heart_disease_data.csv')  # Replace with your dataset file path

# Basic data exploration
print("First few rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())
print("\nMissing values in each column:")
print(df.isnull().sum())

# Data preprocessing
# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = df.copy()
df_imputed[df_imputed.columns] = imputer.fit_transform(df_imputed[df_imputed.columns])

# Encoding categorical variables
label_encoders = {}
categorical_features = df_imputed.select_dtypes(include=['object']).columns
for feature in categorical_features:
    le = LabelEncoder()
    df_imputed[feature] = le.fit_transform(df_imputed[feature])
    label_encoders[feature] = le

# Feature and target selection
X = df_imputed.drop('target', axis=1)
y = df_imputed['target']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction with PCA (optional)
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Model initialization
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Hyperparameter tuning with GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# Grid search for Random Forest
rf_grid = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

# Grid search for Gradient Boosting
gb_grid = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
gb_grid.fit(X_train, y_train)

# Grid search for XGBoost
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Get the best models
rf_best_model = rf_grid.best_estimator_
gb_best_model = gb_grid.best_estimator_
xgb_best_model = xgb_grid.best_estimator_

# Predictions
y_pred_rf = rf_best_model.predict(X_test)
y_pred_gb = gb_best_model.predict(X_test)
y_pred_xgb = xgb_best_model.predict(X_test)

# Evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} - Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

    print(f"\n{model_name} - Classification Report:")
    print(classification_report(y_true, y_pred))

    print(f"\n{model_name} - ROC AUC Score:")
    roc_auc = roc_auc_score(y_true, y_pred)
    print(f"ROC AUC Score: {roc_auc}")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.show()

# Evaluate each model
evaluate_model(y_test, y_pred_rf, "Random Forest")
evaluate_model(y_test, y_pred_gb, "Gradient Boosting")
evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Cross-validation scores
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n{model.__class__.__name__} - Cross-validation Scores:")
    print(f"Scores: {scores}")
    print(f"Mean Score: {scores.mean()}")
    print(f"Standard Deviation: {scores.std()}")

cross_validate_model(rf_best_model, X_pca, y)
cross_validate_model(gb_best_model, X_pca, y)
cross_validate_model(xgb_best_model, X_pca, y)

# Feature importance from the best model
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.figure()
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.show()

plot_feature_importance(xgb_best_model, X.columns)

# Saving the best model (Optional)
import joblib
joblib.dump(rf_best_model, 'rf_best_model.pkl')
joblib.dump(gb_best_model, 'gb_best_model.pkl')
joblib.dump(xgb_best_model, 'xgb_best_model.pkl')

print("\nModels have been saved.")

# Reloading the best model (Optional)
rf_loaded = joblib.load('rf_best_model.pkl')
gb_loaded = joblib.load('gb_best_model.pkl')
xgb_loaded = joblib.load('xgb_best_model.pkl')

print("\nModels have been loaded.")
