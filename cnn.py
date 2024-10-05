import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('/content/sample_data/training_feature.csv')

# Prepare the data: Drop 'name' and target 'ReconciliationRuleId'
X = data.drop(columns=['name', 'ReconciliationRuleId','serialnumber'])
y = data['ReconciliationRuleId']

# One-hot encode the target variable
y = pd.get_dummies(y).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input data to 3D for 1D CNN (samples, timesteps, features)
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

# Build the CNN model
model = Sequential()

# Add a 1D Convolutional layer
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))

# Add a MaxPooling layer (no need to pool too much)
model.add(MaxPooling1D(pool_size=2))

# Add another Convolutional layer with a smaller kernel size to avoid size issues
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))

# Flatten the output without adding another pooling layer
model.add(Flatten())

# Add a Dense fully connected layer
model.add(Dense(128, activation='relu'))

# Add a Dropout layer to reduce overfitting
model.add(Dropout(0.5))

# Add the output layer with softmax for multi-class classification
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Metrics
print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))
print("Confusion Matrix:\n", confusion_matrix(y_test_classes, y_pred_classes))
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes))

# Save model in HDF5 (.h5) format
model.save('cnn_model.h5')
