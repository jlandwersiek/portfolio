## Monte Carlo Techniques for Statistical Analysis of Data

### Overview:  
This project uses Monte Carlo methods for statistical analysis, including calculating the value of π, simulating Poisson and Gaussian distributions, and exploring n-dimensional hypersphere volumes. The methods utilized include acceptance/rejection and transformation techniques.

### Objective:  
- Use Monte Carlo methods to calculate π via the acceptance/rejection method.
- Simulate Poisson deviates and Gaussian deviates using Box-Muller transformation.
- Calculate the volume of n-dimensional hyperspheres using random number sampling.
  
### Project Structure:
```
├── code/  
│   ├── monte_carlo.py # Main script for Monte Carlo simulations  
├── plots/  
│   ├── pi_calculation.png # Plot for calculating π  
│   ├── poisson_deviates.png # Poisson distribution histograms  
│   ├── gaussian_deviates.png # Gaussian deviates generated using Box-Muller  
├── README.md
```

### Technologies Used:
- Python (NumPy, Matplotlib)
- Monte Carlo methods for statistical analysis

### Running the Project:
1. Install required libraries:
   ```
   pip install numpy matplotlib
2. Run the main script:
   ```
   python code/monte_carlo.py
   ```

### Detailed Steps:
#### Step 1: Calculate π using Acceptance/Rejection  
- Random points are chosen within a square, and the ratio of points within a circle gives an estimate for π.  
#### Step 2: Simulate Poisson Deviates  
- Generate histograms for Poisson deviates with different means (μ = 1, μ = 10.3, μ = 102.1).
#### Step 3: Generate Gaussian Deviates using Box-Muller  
- Generate Gaussian deviates using the Box-Muller transformation and calculate χ² values.  
#### Step 4: Calculate n-Dimensional Hypersphere Volumes  
- Use random number sampling to calculate the volume of n-dimensional hyperspheres (3D, 4D, 5D).  

### Results:  
- π Calculation: Converges to 3.1404 with an uncertainty of 0.0016 after sufficient iterations.  
- Poisson Deviates: Histograms for different values of μ show varying concentration around the mean.  
- Gaussian Deviates: χ² values for two perpendicular Gaussian distributions are close to 1.  

### Conclusion:  
Monte Carlo techniques effectively simulate random processes and provide insights into physical sciences, especially in statistical analysis and error estimation.

### References: 
1. P. R. Bevington and D. K. Robinson, Data Reduction and Error Analysis for the Physical Sciences.  
2. J. K. Blitzstein and J. Hwang, Introduction to Probability.  
3. I. G. Hughes and T. P. A. Hase, Measurements and their Uncertainties: A Practical Guide to Modern Error Analysis.  
