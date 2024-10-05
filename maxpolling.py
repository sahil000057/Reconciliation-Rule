from collections import Counter

y_pred_model1 = y_pred_cnn # Predictions from model 1
y_pred_model2 = y_pred_knn  # Predictions from model 2
y_pred_model3 = y_pred_rf  # Predictions from model 3
y_pred_model4 = y_pred_svm # Predictions from model 4
y_pred_model5 = y_pred_tabnet  # Predictions from model 5

# List of predictions for easier iteration
predictions = [y_pred_model1, y_pred_model2, y_pred_model3, y_pred_model4, y_pred_model5]
model_names = ['CNN', 'KNN', 'RF', 'SVM', 'TBN']

# Loop through each model's predictions
for i, y_pred in enumerate(predictions):
    # Count predictions
    pred_counts = dict(Counter(y_pred))
   
    # Sort the prediction counts dictionary by values (counts) in descending order
    sorted_pred_counts = dict(sorted(pred_counts.items(), key=lambda item: item[1], reverse=True))
   
    # Extract keys from the sorted prediction counts dictionary
    sorted_keys = list(sorted_pred_counts.keys())
   
    # Display results for each model
    print(f"Sorted {model_names[i]} Prediction Counts:", sorted_pred_counts)
    print(f"Sorted {model_names[i]} Prediction Keys:", sorted_keys)
    print("\n")  # New line for better readability
