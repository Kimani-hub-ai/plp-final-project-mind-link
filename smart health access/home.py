import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Data
df = pd.read_csv("data/train.csv")

# Encode categorical features
le_location = LabelEncoder()
le_condition = LabelEncoder()

df['Location'] = le_location.fit_transform(df['Location'])
df['Condition'] = le_condition.fit_transform(df['Condition'])

# Feature engineering
df['Income_per_Age'] = df['Income'] / df['Age']

# Features & Target
X = df[['Income', 'Location', 'Age', 'Condition', 'Income_per_Age']]
y = df['HealthAccess']  # Keep as strings for clarity

# Compute class weights for imbalance
classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

# Stratified split to keep class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight=class_weight_dict),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predict & evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print("Best Parameters:", grid_search.best_params_)
print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)
