import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# Modifying the dataset to have more variance in the direction of the new axis (X') and keeping only one class
vec1 = np.array([2, 1]) / np.sqrt(5)
vec2 = np.array([1, 1]) / np.sqrt(2)
# Generating new Gaussian distributed data with more variance in the direction of X'
np.random.seed(0)
class1_x = np.random.normal(5, 3, 50)  # Increased variance
class1_y = np.random.normal(5, 2, 50)
class1 = np.vstack((class1_x, class1_y)).T

# Selecting a subset of data points for projection demonstration
subset_class1 = class1[:40]  # First 10 points

# Calculating the projections on the X' axis (vec1)
proj1_class1 = np.dot(subset_class1, vec1)

# Coordinates of the projected points on the big axis
proj_coords_class1 = np.outer(proj1_class1, vec1) 

full_proj1_class1 = np.dot(class1, vec1)

# Coordinates of the projected points on the big axis
full_proj_coords_class1 = np.outer(full_proj1_class1, vec1)
# Setting up the plot with an increased axis size and no surrounding box
fig, ax1 = plt.subplots(figsize=(10, 6))

# Clear the axis
ax1.cla()

# Plotting the big red axis in the direction of X'
arrow_start = (-2, -1)
arrow_end = (vec1[0] * 15, vec1[1] * 15)
arrow = plt.arrow(*arrow_start, arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1], head_width=0.3, head_length=0.3, fc='r', ec='r', width=0.05)

ax1.add_patch(arrow)

# Plotting the original data points
ax1.scatter(class1_x, class1_y, alpha=0.6, color='blue')

# Plotting the projections of the subset of data points
for point, proj_point in zip(subset_class1, proj_coords_class1):
    ax1.plot([point[0], proj_point[0]], [point[1], proj_point[1]], 'r--', alpha=0.3)  # Line from point to projection
    ax1.scatter(proj_point[0], proj_point[1], color='red', alpha=0.5)

for point, proj_point in zip(class1, full_proj_coords_class1):
    ax1.scatter(proj_point[0], proj_point[1], color='red', alpha=0.5)

# Adding e1 and e2 at the end of the axes
# ax1.annotate('$e_1$', xy=(vec1[0] * 20, vec1[1] * 20), xytext=(vec1[0] * 20, vec1[1] * 20 + 0.5),
#               arrowprops=dict(arrowstyle='->'), fontsize=12, color='r')
# ax1.annotate('$e_2$', xy=(0, 0), xytext=(0.5, 0.5),
#               arrowprops=dict(arrowstyle='->'), fontsize=12, color='r')

# Annotating the projected axis by phi
phi_x = 13  # X-coordinate for the phi annotation
phi_y = 5.9  # Y-coordinate for the phi annotation
ax1.annotate('$\phi_1$', xy=(phi_x, phi_y), fontsize=16, color='r')
ax1.annotate('$e_1$', xy = (13, 3.2), fontsize = 14)
ax1.annotate('$e_2$', xy = (1, 9.2), fontsize = 14)
ax1.axis('off')
ax1.add_patch(FancyArrowPatch((-10, 3), (13, 3), arrowstyle='-|>', mutation_scale=20, color='k'))
ax1.add_patch(FancyArrowPatch((1, -5), (1, 9), arrowstyle='-|>', mutation_scale=20, color='k'))
ax1.set_aspect('equal')

plt.show()

