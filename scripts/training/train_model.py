import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import time
import pickle

input_file = "../../dataset/training/Data.csv"  # Adjust path as needed

method = 'DT'  # Options: 'RF', 'KNN', 'ANN', 'LR', 'DT', 'SVM', 'NB'

# Load data
print("Loading data...")
x = pd.read_csv(input_file, usecols=[2, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26])
y = pd.read_csv(input_file, usecols=[0])
y = np.ravel(y)

print(f"Dataset shape: {x.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Split data: 60% train, 20% validation, 20% test
X_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_temp = scaler.fit_transform(X_temp)
x_test = scaler.transform(x_test)  # Use transform, not fit_transform

train_accuracy = 0
train_precision = 0
train_recall = 0
train_fscore = 0
train_con_matrix = 0

train_time = 0
predict_time = 0

val_accuracy = 0
val_precision = 0
val_recall = 0
val_fscore = 0
val_con_matrix = 0

test_accuracy = 0
test_precision = 0
test_recall = 0
test_fscore = 0
test_con_matrix = 0

print(f"\nTraining {method} model with 10-fold validation...\n")

for i in range(10):
    print(f"Iteration {i+1}/10...")
    
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, train_size=0.75, random_state=i, stratify=y_temp)

    if method == 'RF':
        clf = RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=20, random_state=i)
        
    elif method == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5, weights='distance')

    elif method == 'ANN':
        clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=i)
        
    elif method == 'LR':
        clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=i)

    elif method == 'DT':
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=20, random_state=i)

    elif method == 'SVM':
        clf = LinearSVC(max_iter=1000, random_state=i)

    elif method == 'NB':
        clf = GaussianNB()

    t1 = time.time()    
    clf.fit(X_train, y_train)
    train_time_i = time.time()-t1
    train_time = train_time + train_time_i
    
    y_pred_train = clf.predict(X_train)
    train_accuracy += metrics.accuracy_score(y_train, y_pred_train)
    train_precision += metrics.precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    train_recall += metrics.recall_score(y_train, y_pred_train, average='weighted')
    train_fscore += metrics.f1_score(y_train, y_pred_train, average='weighted')
    train_con_matrix += confusion_matrix(y_train, y_pred_train)

    y_pred_val = clf.predict(X_val)
    val_accuracy += metrics.accuracy_score(y_val, y_pred_val)
    val_precision += metrics.precision_score(y_val, y_pred_val, average='weighted', zero_division=0)
    val_recall += metrics.recall_score(y_val, y_pred_val, average='weighted')
    val_fscore += metrics.f1_score(y_val, y_pred_val, average='weighted')
    val_con_matrix += confusion_matrix(y_val, y_pred_val)

    t2 = time.time()
    y_pred_test = clf.predict(x_test)
    predict_time_i = time.time()-t2
    predict_time = predict_time + predict_time_i

    test_accuracy += metrics.accuracy_score(y_test, y_pred_test)
    test_precision += metrics.precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_recall += metrics.recall_score(y_test, y_pred_test, average='weighted')
    test_fscore += metrics.f1_score(y_test, y_pred_test, average='weighted')
    test_con_matrix += confusion_matrix(y_test, y_pred_test)

# Average results
train_time = train_time / 10
predict_time = predict_time / 10
train_con_matrix = train_con_matrix / 10
val_con_matrix = val_con_matrix / 10
test_con_matrix = test_con_matrix / 10

print("\n" + "="*50)
print(f"RESULTS FOR {method}")
print("="*50)
print(f"\nTraining time: {train_time*1000:.2f} ms")
print(f"Prediction time: {predict_time*1000:.2f} ms")

# Calculate metrics
def calculate_metrics(con_matrix):
    con_matrix = np.ravel(con_matrix)
    
    # Class 0 (Clean)
    TP_1 = con_matrix[0]
    FP_1 = con_matrix[3] + con_matrix[6]
    FN_1 = con_matrix[1] + con_matrix[2]
    TN_1 = con_matrix[4] + con_matrix[8]
    FAR_1 = FP_1/(FP_1+TN_1) if (FP_1+TN_1) > 0 else 0
    MDR_1 = FN_1/(TP_1+FN_1) if (TP_1+FN_1) > 0 else 0
    
    # Class 1 (Static)
    TP_2 = con_matrix[4]
    FP_2 = con_matrix[1] + con_matrix[7]
    FN_2 = con_matrix[3] + con_matrix[5]
    TN_2 = con_matrix[0] + con_matrix[8]
    FAR_2 = FP_2/(FP_2+TN_2) if (FP_2+TN_2) > 0 else 0
    MDR_2 = FN_2/(TP_2+FN_2) if (TP_2+FN_2) > 0 else 0
    
    # Class 2 (Dynamic)
    TP_3 = con_matrix[8]
    FP_3 = con_matrix[2] + con_matrix[5]
    FN_3 = con_matrix[6] + con_matrix[7]
    TN_3 = con_matrix[0] + con_matrix[4]
    FAR_3 = FP_3/(FP_3+TN_3) if (FP_3+TN_3) > 0 else 0
    MDR_3 = FN_3/(TP_3+FN_3) if (TP_3+FN_3) > 0 else 0
    
    FAR = (FAR_1 + FAR_2 + FAR_3) / 3
    MDR = (MDR_1 + MDR_2 + MDR_3) / 3
    
    return FAR, MDR

FAR_train, MDR_train = calculate_metrics(train_con_matrix)
FAR_val, MDR_val = calculate_metrics(val_con_matrix)
FAR_test, MDR_test = calculate_metrics(test_con_matrix)

train_score = np.array([train_accuracy/10, train_precision/10, train_recall/10, train_fscore/10, FAR_train, MDR_train])
val_score = np.array([val_accuracy/10, val_precision/10, val_recall/10, val_fscore/10, FAR_val, MDR_val])
test_score = np.array([test_accuracy/10, test_precision/10, test_recall/10, test_fscore/10, FAR_test, MDR_test])

# Display results
columns = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'FAR', 'MDR']
index = ['Training', 'Validation', 'Testing']
results_df = pd.DataFrame(np.array([train_score, val_score, test_score]), index=index, columns=columns)

print("\n" + str(results_df))
print("\n")

# Save results
results_df.to_csv(method + "_score.csv", index=True, header=True)
print(f"Results saved to {method}_score.csv")

# Save the final model
with open(method + '_model.pkl', 'wb') as f:
    pickle.dump((clf, scaler), f)
print(f"Model saved to {method}_model.pkl")

# Plot confusion matrices
target_names = ['Clean', 'Static', 'Dynamic']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ConfusionMatrixDisplay.from_estimator(clf, X_val, y_val, display_labels=target_names, cmap="GnBu")
plt.title(f'{method} Validation Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{method}_validation_confusion_matrix.png', dpi=300, bbox_inches='tight')

plt.figure()
ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test, display_labels=target_names, cmap="GnBu")
plt.title(f'{method} Testing Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{method}_testing_confusion_matrix.png', dpi=300, bbox_inches='tight')

print(f"Confusion matrices saved")
plt.show()
