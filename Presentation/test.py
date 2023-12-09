import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

# Function to handle click events
def onclick(event):
    global vec1, vec2
    # Update vec1 based on click position
    vec1 = np.array([event.xdata, event.ydata]) - np.array([6, 8])
    vec1 /= np.linalg.norm(vec1)
    # Calculate orthogonal vector
    vec2 = np.array([-vec1[1], vec1[0]])

    # Redraw the plot
    draw_plot()

# Function to draw the plot
def draw_plot():
    ax1.cla()  # Clear the axis
    ax2.cla()  # Clear the axis
    ax3.cla()  # Clear the axis

    # Re-projecting the data
    proj1_class1 = np.dot(class1, vec1)
    proj1_class2 = np.dot(class2, vec1)
    proj2_class1 = np.dot(class1, vec2)
    proj2_class2 = np.dot(class2, vec2)

    # Re-plotting the data and arrows on the main plot
    ax1.scatter(class1_x, class1_y, alpha=0.6, label='Class 1')
    ax1.scatter(class2_x, class2_y, alpha=0.6, label='Class 2')
    ax1.arrow(6, 8, vec1[0]*2, vec1[1]*2, head_width=0.3, head_length=0.3, fc='k', ec='k', width=0.05)
    ax1.arrow(6, 8, vec2[0]*2, vec2[1]*2, head_width=0.3, head_length=0.3, fc='k', ec='k', width=0.05)
    ax1.text(4.5, 2, 'X', ha='center', va='center', fontweight = 'bold')
    ax1.text(2, 4.5, 'Y', ha='center', va='center', fontweight = 'bold')
    ax1.arrow(2, 2, 2, 0, head_width=0.3, head_length=0.3, fc='k', ec='k', width=0.05)
    ax1.arrow(2, 2, 0, 2, head_width=0.3, head_length=0.3, fc='k', ec='k', width=0.05)
    # Redrawing other elements (axes, titles, etc.)

    ax1.text(6 + vec1[0]*2+.5, 8 + vec1[1]*2+.5, 'X\'', ha='center', va='center', color='k',fontweight = 'bold')
    ax1.text(6 +vec2[0]*2 +.5, 8 + vec2[1]*2+.5, 'Y\'', ha='center', va='center', color='k' ,fontweight = 'bold')

    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Random Distributions', fontsize=14)
    ax1.set_xlabel('X Value')
    ax1.set_ylabel('Y Value')
    ax1.legend()

    # Updating the histograms in the subplots
    ax2.hist(proj1_class1, bins=30, alpha=0.6, label='Class 1')
    ax2.hist(proj1_class2, bins=30, alpha=0.6, label='Class 2')
    # ax2.set_title('Data along X\' Axis', color = "k", fontsize = 12)
    ax2.set_xlabel('X\'', color = "k", fontweight = 'bold')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    ax3.hist(proj2_class1, bins=30, alpha=0.6, label='Class 1')
    ax3.hist(proj2_class2, bins=30, alpha=0.6, label='Class 2')
    # ax3.set_title('Data along Y\' Axis', color = "k", fontsize = 12)
    ax3.set_xlabel('Y\'', color = "k", fontweight = 'bold')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    fig.canvas.draw_idle()  # Update the figure

# Generating the Gaussian distributed data
np.random.seed(0)
class1_x = np.random.normal(5, 1, 300)
class1_y = np.random.normal(5, 1, 300)
class2_x = np.random.normal(8, 1.5, 300)
class2_y = np.random.normal(10, 1.5, 300)

# Combining x and y coordinates
class1 = np.vstack((class1_x, class1_y)).T
class2 = np.vstack((class2_x, class2_y)).T

# Initial direction vectors
vec1 = np.array([1, 1]) / np.sqrt(2)  # Direction of first arrow
vec2 = np.array([-1, 1]) / np.sqrt(2)  # Orthogonal direction

# Creating the main plot and subplots
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 1))

# Draw the initial plot
draw_plot()

# Connect the click event to the handler
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
