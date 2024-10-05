import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from scipy import stats  # For majority voting

# Load the dataset
data = pd.read_csv(r'/content/sample_data/training_feature.csv')

# Prepare the data: Drop 'name', 'serialnumber', and target 'ReconciliationRuleId'
X = data.drop(columns=['name', 'ReconciliationRuleId', 'serialnumber'])
y = data['ReconciliationRuleId']

# Convert target variable to integer labels for consistency across models
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for all models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Model 1: SVM
# --------------------------
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# --------------------------
# Model 2: KNN
# --------------------------
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

# --------------------------
# Model 3: TabNet
# --------------------------
tabnet_model = TabNetClassifier()
tabnet_model.fit(X_train_scaled, y_train, max_epochs=100, patience=20, batch_size=32)
y_pred_tabnet = tabnet_model.predict(X_test_scaled)

# --------------------------
# Model 4: CNN
# --------------------------
# Reshape input data to 3D for CNN (samples, timesteps, features)
X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

# One-hot encode the target variable for CNN
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)

# Build CNN model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(y_train_cnn.shape[1], activation='softmax'))

# Compile and train the CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test_cnn))

# Make predictions with CNN
y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)

# --------------------------
# Model 5: Random Forest
# --------------------------
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# --------------------------
# Max Polling (Majority Voting)
# --------------------------
# Collect predictions from all models
predictions = np.vstack((y_pred_svm, y_pred_knn, y_pred_tabnet, y_pred_cnn, y_pred_rf)).T

# Perform majority voting
y_pred_final = stats.mode(predictions, axis=1)[0].flatten()

# --------------------------
# Evaluation of Final Model (after max polling)
# --------------------------
print("Final Model (Max Polling):")
print("Accuracy:", accuracy_score(y_test, y_pred_final))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))
print("Classification Report:\n", classification_report(y_test, y_pred_final))

# --------------------------
# Evaluate Each Individual Model (Optional)
# --------------------------
print("\nIndividual Model Performance:")

print("\nSVM Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

print("\nKNN Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

print("\nTabNet Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_tabnet))
print("Classification Report:\n", classification_report(y_test, y_pred_tabnet))

print("\nCNN Model:")
print("Accuracy:", accuracy_score(np.argmax(y_test_cnn, axis=1), y_pred_cnn))
print("Classification Report:\n", classification_report(np.argmax(y_test_cnn, axis=1), y_pred_cnn))

print("\nRandom Forest Model:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

