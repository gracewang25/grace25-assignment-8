import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Function to generate ellipsoid clusters
def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5, direction=np.array([1, 1])):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                   [cluster_std * 0.8, cluster_std]])
    
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1) and shift along the direction
    shift = distance * direction
    X2 = np.random.multivariate_normal(mean=[1 + shift[0], 1 + shift[1]], cov=covariance_matrix, size=n_samples)
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def do_experiments(start, end, step_num):
    # Set up experiment parameters
    shift_distances = np.linspace(start, end, step_num)  # Range of shift distances
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []

    # Parameters for dataset visualization
    n_cols = 2
    n_rows = int(np.ceil(len(shift_distances) / n_cols))  # Calculate rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 10))  # Create subplots

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Run experiments for each shift distance
    for i, distance in enumerate(shift_distances):
        ax = axes[i]  # Get the current subplot
        # Generate dataset (shift along y = -x direction)
        X, y = generate_ellipsoid_clusters(distance=distance, direction=np.array([-1, 1]))

        # Fit logistic regression using existing function
        model = LogisticRegression()
        model.fit(X, y)
        beta0 = model.intercept_[0]
        beta1, beta2 = model.coef_[0]
        slope = -beta1 / beta2
        intercept = -beta0 / beta2

        # Record logistic regression parameters
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope_list.append(slope)
        intercept_list.append(intercept)

        # Compute logistic loss
        y_prob = model.predict_proba(X)
        logistic_loss = log_loss(y, y_prob)
        loss_list.append(logistic_loss)

        # Calculate margin width
        x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
        y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        # Contour levels and fading effects
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]  # Fading effect for confidence levels
        min_distance = np.nan  # Default value for missing contours
        for level, alpha in zip(contour_levels, alphas):
            # Restrict contour plotting to the current axes
            class_1_contour = ax.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            class_0_contour = ax.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
            if level == 0.7 and (len(class_1_contour.collections) > 0 and len(class_0_contour.collections) > 0):
                distances = cdist(class_1_contour.collections[0].get_paths()[0].vertices,
                                  class_0_contour.collections[0].get_paths()[0].vertices, metric='euclidean')
                min_distance = np.min(distances)
        margin_widths.append(min_distance)

        # Plot the dataset
        ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0', alpha=0.8)
        ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1', alpha=0.8)
        x_vals = np.linspace(x_min, x_max, 200)
        ax.plot(x_vals, slope * x_vals + intercept, color='black', linestyle='--', label='Decision Boundary')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Shift Distance = {distance:.2f}", fontsize=12)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if i == 0:
            ax.legend(loc='lower right', fontsize=10)

        # Add equation and margin width text within the subplot
        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        margin_text = f"Margin Width: {min_distance:.2f}"
        ax.text(x_min + 0.5, y_max - 0.5, equation_text, fontsize=10, color="black", ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        ax.text(x_min + 0.5, y_max - 1.5, margin_text, fontsize=10, color="black", ha='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Turn off any unused axes (in case of uneven grid)
    for j in range(len(shift_distances), len(axes)):
        axes[j].axis('off')

    # Adjust layout and save dataset visualization
    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")

    # Plot parameter trends
    plt.figure(figsize=(18, 15))

    # Plot beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")
    plt.grid(True)

    # Plot beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")
    plt.grid(True)

    # Plot beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")
    plt.grid(True)

    # Plot slope (beta1 / beta2)
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o')
    plt.title("Shift Distance vs Slope (Beta1 / Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")
    plt.grid(True)

    # Plot intercept (beta0 / beta2)
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_list, marker='o')
    plt.title("Shift Distance vs Intercept (Beta0 / Beta2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Intercept")
    plt.grid(True)

    # Plot logistic loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")
    plt.grid(True)

    # Plot margin width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, marker='o')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")
    plt.grid(True)

    # Adjust layout and save parameter trends
    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
