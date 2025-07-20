import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import os
import seaborn as sns

# Download penguins dataset
df = sns.load_dataset("penguins")

print("Initial dataset shape:", df.shape)

df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

df = pd.get_dummies(df, columns=["sex", "island"])

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("Training set class distribution:")
print(y_train.value_counts())
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    use_label_encoder=False,
    eval_metric="mlogloss",
    verbosity=0,
)
model.fit(X_train, y_train)

print("Model training complete.")
# Print class info
n_classes = len(le.classes_)
print(f"Number of classes: {n_classes}")
print(f"Class names: {list(le.classes_)}")

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("\nTrain classification report:")
print(classification_report(y_train, train_pred, target_names=le.classes_))
print("\nTest classification report:")
print(classification_report(y_test, test_pred, target_names=le.classes_))

print("\nTest confusion matrix:")
print(confusion_matrix(y_test, test_pred))

# Save model to app/data/model.json
os.makedirs("app/data", exist_ok=True)
model.save_model("app/data/model.json")
print("Model trained and saved to app/data/model.json")
