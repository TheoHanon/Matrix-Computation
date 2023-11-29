import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import matplotlib.lines as mlines
# Re-generating the Gaussian distributed data
np.random.seed(0)
class1_x = np.random.normal(5, 1, 300)
class1_y = np.random.normal(5, 1, 300)
class2_x = np.random.normal(10, 2, 300)
class2_y = np.random.normal(10, 2, 300)

# Combining x and y coordinates
class1 = np.vstack((class1_x, class1_y)).T
class2 = np.vstack((class2_x, class2_y)).T

# Define the direction vectors for the custom axes
vec1 = np.array([1, 1]) / np.sqrt(2)  # Direction of first arrow
vec2 = np.array([-1, 1]) / np.sqrt(2) # Direction of second arrow

# Project the data onto the custom axes
proj1_class1 = np.dot(class1, vec1)
proj1_class2 = np.dot(class2, vec1)
proj2_class1 = np.dot(class1, vec2)
proj2_class2 = np.dot(class2, vec2)

# Creating the main plot and subplots
fig = plt.figure(figsize=(12, 6))

# Main plot
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax1.scatter(class1_x, class1_y, alpha=0.6, label='Class 1')
ax1.scatter(class2_x, class2_y, alpha=0.6, label='Class 2')
ax1.arrow(9, 9, 2, 2, head_width=0.3, head_length=0.3, fc='k', ec='k', width=0.05)
ax1.arrow(9, 9, -2, 2, head_width=0.3, head_length=0.3, fc='grey', ec='grey', width=0.05)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title('Random Distributions', fontsize = 14)
ax1.set_xlabel('X Value')
ax1.set_ylabel('Y Value')





# Create proxy artists for the arrows in the legend
arrow_proxy1 = mlines.Line2D([], [], color='k', marker='>', markeredgewidth=2, 
                             linestyle='none', markersize=4, label='New Axis 1')
arrow_proxy2 = mlines.Line2D([], [], color='grey', marker='>', markeredgewidth=2, 
                             linestyle='none', markersize=4, label='New Axis 2')

# Adding these proxies to the existing legend
# First, get the current handles and labels
current_handles, current_labels = ax1.get_legend_handles_labels()

# Add the new proxy legends to the list of handles and labels
current_handles.extend([arrow_proxy1, arrow_proxy2])
current_labels.extend(['Custom Axis 1', 'Custom Axis 2'])

# Create the new legend
ax1.legend(handles=current_handles, labels=current_labels)



ax1.grid(True)

# Subplot for projection along the first custom axis
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.hist(proj1_class1, bins=30, alpha=0.6, label='Class 1')
ax2.hist(proj1_class2, bins=30, alpha=0.6, label='Class 2')
ax2.set_title('Data along New Axis 1', color = "k", fontsize = 12)
ax2.set_xlabel('New axis 1')
ax2.set_ylabel('Frequency')
ax2.legend()

# Subplot for projection along the second custom axis
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax3.hist(proj2_class1, bins=30, alpha=0.6, label='Class 1')
ax3.hist(proj2_class2, bins=30, alpha=0.6, label='Class 2')
ax3.set_title('Data along New Axis 2', color = "grey", fontsize = 12)
ax3.set_xlabel('New axis 2',)
ax3.set_ylabel('Frequency')
ax3.legend()

plt.tight_layout()
plt.show()
