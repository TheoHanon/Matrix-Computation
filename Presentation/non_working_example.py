import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons, make_circles, make_classification

# Generate some sample data
X, y = make_moons(n_samples=100, random_state=42)

# Define the list of kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

# Create subplots for each kernel
fig, axs = plt.subplots(2, len(kernels)+1, figsize=(15, 6))

# Plot the initial data
axs[0, 0].scatter(X[:, 0], X[:, 1], c=y)
axs[0, 0].set_xlabel('Feature 1')
axs[0, 0].set_ylabel('Feature 2')
axs[0, 0].set_title('Initial Data')

axs[1, 0].scatter(X[:, 0], X[:, 1], c=y)
axs[1, 0].set_xlabel('Feature 1')
axs[1, 0].set_ylabel('Feature 2')
axs[1, 0].set_title('Initial Data')

for i, kernel in enumerate(kernels):
    # Create KernelPCA object with k principal components
    k = 1  # Number of principal components
    kpca = KernelPCA(n_components=k, kernel=kernel)

    # Fit the data to the KernelPCA model
    X_kpca = kpca.fit_transform(X)

    # Plot the principal components
    axs[0, i+1].scatter(X_kpca[:, 0], np.zeros_like(X_kpca[:, 0]), c=y)
    axs[0, i+1].set_xlabel('Principal Component 1')
    axs[0, i+1].set_ylabel('Zero')
    axs[0, i+1].set_title('Kernel: {}'.format(kernel))

# Iterate over each kernel and plot the results
for i, kernel in enumerate(kernels):
    # Create KernelPCA object with k principal components
    k = 2  # Number of principal components
    kpca = KernelPCA(n_components=k, kernel=kernel)

    # Fit the data to the KernelPCA model
    X_kpca = kpca.fit_transform(X)

    # Plot the principal components
    axs[1, i+1].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
    axs[1, i+1].set_xlabel('Principal Component 1')
    axs[1, i+1].set_ylabel('Principal Component 2')
    axs[1, i+1].set_title('Kernel: {}'.format(kernel))

plt.tight_layout()
plt.show()
