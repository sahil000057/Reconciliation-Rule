import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('/content/sample_data/training_feature.csv')

# Prepare the data: Drop 'name' and target 'ReconciliationRuleId'
X = data.drop(columns=['name', 'ReconciliationRuleId','serialnumber'])
y = data['ReconciliationRuleId']

# Convert target variable to integer labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TabNet model initialization
model = TabNetClassifier()

# Train the model
model.fit(X_train_scaled, y_train, max_epochs=100, patience=20, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test_scaled)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


model.save_model('tabnet_model')