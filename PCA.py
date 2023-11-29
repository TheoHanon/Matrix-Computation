import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 1 component
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
y_numeric = np.reshape(np.array(y), (569, ))
y_numeric[y_numeric == 'M'] = 1
y_numeric[y_numeric == 'B'] = 0
# Apply linear classifier (Logistic Regression for binary classification)
classifier = LogisticRegression()
classifier.fit(X_pca, y)

# Create a scatter plot for the one-dimensional PCA data
plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c = y_numeric,cmap='viridis', edgecolor='k', s=20)

# Plot the decision boundary
decision_boundary = -classifier.intercept_ / classifier.coef_[0]
plt.axvline(x=decision_boundary, color='red', linestyle='--')

plt.xlabel('PCA Component 1')
plt.ylim([-0.1, 0.1])
plt.title('1D PCA with Logistic Regression Decision Boundary')
plt.yticks([])
plt.show()




# Continuing from the provided code, let's now apply PCA with 2 components and a new classifier

# Apply PCA with 2 components
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

# Convert y to numeric format if it's not already
y_numeric = np.array(y)
y_numeric[y_numeric == 'M'] = 1
y_numeric[y_numeric == 'B'] = 0
y_numeric = y_numeric.astype(int)

# Apply linear classifier (Logistic Regression for binary classification) on the 2-component PCA data
classifier_2 = LogisticRegression()
classifier_2.fit(X_pca_2, y)

# Create a scatter plot for the two-dimensional PCA data
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=y_numeric, cmap='viridis', edgecolor='k', s=20)

# Highlight the decision boundary
# For 2D PCA, the decision boundary is a line, not a single point
coef = classifier_2.coef_[0]
intercept = classifier_2.intercept_
x_values = np.linspace(X_pca_2[:, 0].min(), X_pca_2[:, 0].max(), 100)
# Decision boundary line: coef[0]*x + coef[1]*y + intercept = 0
# => y = -(intercept + coef[0]*x) / coef[1]
y_values = -(intercept + coef[0]*x_values) / coef[1]
plt.plot(x_values, y_values, color='red', linestyle='--')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA with Logistic Regression Decision Boundary')
plt.show()



pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)

# Apply linear classifier (Logistic Regression for binary classification) on the 3-component PCA data
classifier_3 = LogisticRegression()
classifier_3.fit(X_pca_3, y_numeric)

coefficients = classifier_3.coef_[0]
intercept = classifier_3.intercept_

xx, yy = np.meshgrid(np.linspace(X_pca_3[:, 0].min(), X_pca_3[:, 0].max(), 100),
                     np.linspace(X_pca_3[:, 1].min(), X_pca_3[:, 1].max(), 100))

# Calculate corresponding z for the plane
zz = -(intercept + coefficients[0]*xx + coefficients[1]*yy) / coefficients[2]

# 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=y_numeric, cmap='viridis', edgecolor='k', s=50)

# Changing the color of the decision plane
# Let's choose a blue color with a slight transparency
surface = ax.plot_surface(xx, yy, zz, alpha=0.3, color='skyblue', edgecolor='none')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.title('3D PCA with Logistic Regression Decision Hyperplane (Color Adjusted)')
plt.show()




import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt

# Fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

# Data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets

# Convert y to numeric format
y_numeric = np.array(y)
y_numeric[y_numeric == 'M'] = 1
y_numeric[y_numeric == 'B'] = 0
y_numeric = y_numeric.astype(int)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to evaluate classifier
def evaluate_classifier(classifier, X_data, y_data):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = classifier.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1, auc

# Apply PCA with 1, 2, and 3 components
pca_1 = PCA(n_components=1)
X_pca_1 = pca_1.fit_transform(X_scaled)

pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)

# Initialize classifiers
classifier_1 = LogisticRegression()
classifier_2 = LogisticRegression()
classifier_3 = LogisticRegression()

# Evaluate classifiers
metrics_1 = evaluate_classifier(classifier_1, X_pca_1, y_numeric)
metrics_2 = evaluate_classifier(classifier_2, X_pca_2, y_numeric)
metrics_3 = evaluate_classifier(classifier_3, X_pca_3, y_numeric)

# Metrics names
# Adjusting the code to remove F1 score and AUC, and create a single bar chart for all classifiers

# Filter out F1 Score and AUC from the metrics
metrics_1_filtered = metrics_1[:3]
metrics_2_filtered = metrics_2[:3]
metrics_3_filtered = metrics_3[:3]

# Updated metrics names (without F1 Score and AUC)
updated_metrics_names = ['Accuracy', 'Precision', 'Recall']

# Data for plotting
data_for_plotting_filtered = [metrics_1_filtered, metrics_2_filtered, metrics_3_filtered]

# Creating a single bar chart for all classifiers
fig, ax = plt.subplots(figsize=(10, 6))

# Set position of bar on X axis
barWidth = 0.25
r1 = np.arange(len(metrics_1_filtered))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
ax.bar(r1, metrics_1_filtered, color='skyblue', width=barWidth, edgecolor='grey', label='1D PCA')
ax.bar(r2, metrics_2_filtered, color='orange', width=barWidth, edgecolor='grey', label='2D PCA')
ax.bar(r3, metrics_3_filtered, color='green', width=barWidth, edgecolor='grey', label='3D PCA')

# Add labels and title
ax.set_xlabel('Metrics', fontweight='bold', fontsize=15)
ax.set_ylabel('Scores', fontweight='bold', fontsize=15)
ax.set_title('Comparison of Classifier Performance (1D, 2D, 3D PCA)', fontweight='bold', fontsize=16)
ax.set_xticks([r + barWidth for r in range(len(metrics_1_filtered))])
ax.set_xticklabels(updated_metrics_names)

# Create legend & Show graphic
ax.legend()
plt.show()


