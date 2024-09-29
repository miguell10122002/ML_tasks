import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your synthetic data (replace 'X_data.npy' and 'y_data.npy' with your data files)
X = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')

# Get the number of features (columns) in X
num_features = X.shape[1]

# Create histograms for each feature in X
for i in range(num_features):
    plt.figure(figsize=(8, 6))
    sns.histplot(X[:, i], bins=20, kde=True)
    plt.xlabel(f'Feature {i + 1}')
    plt.ylabel('Frequency')
    plt.title(f'Histogram for Feature {i + 1}')

plt.show()
# Create a histogram for the target variable y
plt.figure(figsize=(8, 6))
sns.histplot(y, bins=20, kde=True)
plt.xlabel('Target (y)')
plt.ylabel('Frequency')
plt.title('Histogram for Target (y)')
plt.show()
