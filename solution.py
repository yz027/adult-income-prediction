import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------------------- Data Loading and Initial Exploration ----------------------
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

train_data = pd.read_csv("adult.data", header=None, names=columns, na_values=" ?", skipinitialspace=True)
test_data = pd.read_csv("adult.test", skiprows=1, header=None, names=columns, na_values=" ?", skipinitialspace=True)

# Remove missing values
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# Convert income to binary labels
train_data['income'] = train_data['income'].apply(lambda x: 1 if '>50K' in x else 0)
test_data['income'] = test_data['income'].apply(lambda x: 1 if '>50K' in x else 0)

# Remove irrelevant feature
train_data.drop(columns=['fnlwgt'], inplace=True)
test_data.drop(columns=['fnlwgt'], inplace=True)

# One-hot encoding
def encode_data(data):
    return pd.get_dummies(data, drop_first=True)

train_data = encode_data(train_data)
test_data = encode_data(test_data)

test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# ---------------------- Feature Selection and Data Splitting ----------------------
X = train_data.drop('income', axis=1)
y = train_data['income']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ---------------------- Model Training and Hyperparameter Tuning ----------------------
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# ---------------------- Model Evaluation ----------------------
y_pred = best_model.predict(X_val_scaled)
print(classification_report(y_val, y_pred))
ConfusionMatrixDisplay.from_estimator(best_model, X_val_scaled, y_val)
plt.show()

# ---------------------- Feature Importance Visualization ----------------------
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'][:15], feature_importance_df['Importance'][:15])
plt.xlabel('Feature Importance')
plt.title('Top 15 Important Features')
plt.gca().invert_yaxis()
plt.show()

