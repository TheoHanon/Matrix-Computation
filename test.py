import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import matplotlib.lines as mlines
# Re-generating the Gaussian distributed data
np.random.seed(0)
class1_x = np.random.normal(5, 1, 300)
class1_y = np.random.normal(5, 1, 300)
class2_x = np.random.normal(8, 1.5, 300)
class2_y = np.random.normal(10, 1.5, 300)

# Combining x and y coordinates
class1 = np.vstack((class1_x, class1_y)).T
class2 = np.vstack((class2_x, class2_y)).T

# Define the direction vectors for the custom axes
vec1 = np.array([1, 1]) / np.sqrt(2)  # Direction of first arrow
vec2 = np.array([1, 0]) / np.sqrt(2) # Direction of second arrow

# Project the data onto the custom axes
proj1_class1 = np.dot(class1, vec1)
proj1_class2 = np.dot(class2, vec1)
proj2_class1 = np.dot(class1, vec2)
proj2_class2 = np.dot(class2, vec2)

# Creating the main plot and subplots
fig = plt.figure(figsize=(10, 5))

# Main plot
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax1.grid(True, alpha = 0.2)
ax1.scatter(class1_x, class1_y, alpha=0.6, label='Class 1')
ax1.scatter(class2_x, class2_y, alpha=0.6, label='Class 2')
ax1.arrow(6, 8, np.sqrt(2), np.sqrt(2), head_width=0.3, head_length=0.3, fc= 'k', ec='k', width=0.05)
ax1.arrow(6, 8, -np.sqrt(2), np.sqrt(2), head_width=0.3, head_length=0.3, fc='k', ec='k', width=0.05)
ax1.arrow(2, 2, 2, 0, head_width=0.3, head_length=0.3, fc='k', ec='k', width=0.05)
ax1.arrow(2, 2, 0, 2, head_width=0.3, head_length=0.3, fc='k', ec='k', width=0.05)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title('Random Distributions', fontsize = 14)
ax1.set_xlabel('X Value')
ax1.set_ylabel('Y Value')





# # Create proxy artists for the arrows in the legend
# arrow_proxy1 = mlines.Line2D([], [], color='k', marker='>', markeredgewidth=2, 
#                              linestyle='none', markersize=4, label='New Axis')
# arrow_proxy2 = mlines.Line2D([], [], color='k', marker='>', markeredgewidth=2, 
#                              linestyle='none', markersize=4, label='Old Axis')

# # Adding these proxies to the existing legend
# # First, get the current handles and labels
# current_handles, current_labels = ax1.get_legend_handles_labels()

# # Add the new proxy legends to the list of handles and labels
# current_handles.extend([arrow_proxy1, arrow_proxy2])
# current_labels.extend(['New Axis', 'Old Axis'])

# Annotating the old axes
ax1.text(4.5, 2, 'X', ha='center', va='center', fontweight = 'bold')
ax1.text(2, 4.5, 'Y', ha='center', va='center', fontweight = 'bold')

# Annotating the new axes
ax1.text(6 + np.sqrt(2)+.5, 8 + np.sqrt(2)+.5, 'X\'', ha='center', va='center', color='k',fontweight = 'bold')
ax1.text(6 -np.sqrt(2)-.5, 8 + np.sqrt(2)+.5, 'Y\'', ha='center', va='center', color='k' ,fontweight = 'bold')

# Create the new legend
ax1.legend()





# Subplot for projection along the first custom axis
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.hist(proj1_class1, bins=30, alpha=0.6, label='Class 1')
ax2.hist(proj1_class2, bins=30, alpha=0.6, label='Class 2')
ax2.set_title('Data along New Axis', color = "k", fontsize = 12)
ax2.set_xlabel('X\'', color = "k", fontweight = 'bold')
ax2.set_ylabel('Frequency')
ax2.legend()

# Subplot for projection along the second custom axis
ax3 = plt.subplot2grid((2, 2), (1, 1))
ax3.hist(proj2_class1, bins=30, alpha=0.6, label='Class 1')
ax3.hist(proj2_class2, bins=30, alpha=0.6, label='Class 2')
ax3.set_title('Data along X-Axis', color = "k", fontsize = 12)
ax3.set_xlabel('X',color = "k", fontweight = 'bold')
ax3.set_ylabel('Frequency')
ax3.legend()

plt.tight_layout()
plt.show()
