import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Pi using the acceptance/rejection method
def estimate_pi(num_points):
    inside_circle = 0
    for _ in range(num_points):
        x, y = np.random.uniform(-1, 1, 2)
        if x**2 + y**2 <= 1:
            inside_circle += 1
    pi_estimate = 4 * inside_circle / num_points
    return pi_estimate

# Function to calculate volume of n-dimensional hyperspheres using acceptance/rejection
def hypersphere_volume(n, num_points):
    hits = 0
    for _ in range(num_points):
        point = np.random.uniform(-1, 1, n)
        if np.sum(point**2) <= 1:
            hits += 1
    volume_estimate = (2 ** n) * hits / num_points
    return volume_estimate

# Function to generate Poisson deviates
def poisson_deviates(mu, num_samples):
    deviates = []
    for _ in range(num_samples):
        L = np.exp(-mu)
        k = 0
        p = 1
        while p > L:
            k += 1
            p *= np.random.uniform(0, 1)
        deviates.append(k-1)
    return deviates

# Box-Muller Transformation to generate Gaussian deviates
def box_muller(num_samples):
    u1 = np.random.uniform(0, 1, num_samples)
    u2 = np.random.uniform(0, 1, num_samples)
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2

# Plotting functions for the histograms of Poisson deviates
def plot_poisson_histogram(mu, num_samples):
    deviates = poisson_deviates(mu, num_samples)
    plt.hist(deviates, bins=30, density=True, alpha=0.6, color='g')
    plt.title(f'Poisson Distribution with μ = {mu}')
    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.show()

# Plotting functions for the Gaussian deviates
def plot_gaussian_histogram(num_samples):
    z1, z2 = box_muller(num_samples)
    plt.hist(z1, bins=30, density=True, alpha=0.6, color='b')
    plt.title('Box-Muller Gaussian Distribution')
    plt.xlabel('Gaussian Deviates (x)')
    plt.ylabel('Frequency')
    plt.show()
    plt.hist(z2, bins=30, density=True, alpha=0.6, color='b')
    plt.title('Box-Muller Gaussian Distribution')
    plt.xlabel('Gaussian Deviates (y)')
    plt.ylabel('Frequency')
    plt.show()
# Running the code to demonstrate the methods

# Estimating Pi
num_points = 1000000
pi_estimate = estimate_pi(num_points)
print(f"Estimated value of Pi with {num_points} points: {pi_estimate}")

# Estimating the volume of 3, 4, and 5-dimensional hyperspheres
for n in range(3, 6):
    volume_estimate = hypersphere_volume(n, 10000)
    print(f"Estimated volume of {n}-dimensional hypersphere: {volume_estimate}")

# Plotting Poisson deviates histograms for different mu values
for mu in [1, 10.3, 102.1]:
    plot_poisson_histogram(mu, 10000)

# Plotting Gaussian deviates using the Box-Muller transformation
plot_gaussian_histogram(10000)
