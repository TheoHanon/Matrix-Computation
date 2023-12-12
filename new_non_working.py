from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
import numpy as np

# Step 1: Generate Data
X, y = make_circles(n_samples=400, factor=.3, noise=.05)

# Step 2: Apply Kernel PCA
kpca = KernelPCA(kernel="rbf", n_components=3)
X_kpca = kpca.fit_transform(X)

clf = SVC()
clf.fit(X_kpca, y)
print(clf.score(X_kpca, y))

# Step 3: Visualize the 3D Embedding
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2], c=y)
ax.set_title("3D Embedding using Kernel PCA")
plt.show()
