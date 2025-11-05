# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv("npc_behavior_dataset_modified.csv") 

# Separate features and label 
X = df.drop("NPC_Action_Type", axis=1)
y = df["NPC_Action_Type"] 

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col]) 
    label_encoders[col] = le

# Encode target label
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
model.fit(X_train, y_train)
                         
# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)  
print(f" Model Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45) 
plt.title("NPC Action Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Save model and encoders
joblib.dump(model, "npc_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("\n Model, encoders, and confusion matrix saved successfully!")
