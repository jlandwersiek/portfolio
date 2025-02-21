## Power Grid Stability and Forecasting  

### Overview:  
This project simulates a basic power grid network, generates demand fluctuations for substations, and applies statistical and machine learning techniques to analyze the grid's stability. The project also forecasts future power demand and classifies substations into stability categories based on their demand fluctuations.

### Objective:  
- Model a power grid with substations, transmission lines, and power demand.
- Simulate demand fluctuations and visualize power grid structure.
- Analyze grid stability using statistical methods.
- Apply machine learning (K-means clustering) to classify substations' stability.
- Forecast future power demand using ARIMA models.
- Visualize the power grid structure, stability analysis, and demand fluctuations.

### Project Structure:
```
├── code/
│   ├── power_grid_analysis.py  # Main script for grid simulation, stability analysis, and forecasting
├── plots/
│   ├── grid_structure.png  # Visualization of the power grid structure
│   ├── substation_stability.png  # Stability classification of substations
│   ├── demand_fluctuations.png  # Demand fluctuations over 24 hours for each substation
│   ├── stability_clustering.png  # K-means clustering of stability categories
└── README.md
```


### Technologies Used:
- Python (NumPy, Pandas, NetworkX, Matplotlib, Scikit-Learn, pmdarima)
- Data Visualization (Matplotlib, NetworkX)
- Machine Learning (K-Means Clustering, ARIMA for forecasting)
- Statistical Analysis (Time-Series Analysis, Load Fluctuation, and Voltage Deviation)
- Running the Project
  
### To run this project:  
  1. Install the required libraries:
```
pip install numpy pandas networkx matplotlib sklearn pmdarima
```
  2. Run the main script:
```
python code/power_grid_stability.py
```
  3. The script will generate visualizations of the grid structure, power demand fluctuations, stability classifications, and clustering results.

### Detailed Steps:
#### Step 1: Simulate a Simple Power Grid  
- Define the grid as a graph with substations as nodes and transmission lines as edges.  
- Simulate hourly power demand fluctuations for each substation over 24 hours.  
- Visualize the power grid structure.  
#### Step 2: Analyze Stability Using Statistical Methods  
- Calculate voltage deviations (standard deviation of demand) and load fluctuations (difference in demand between consecutive hours).  
- Identify substations with significant instability using a threshold for voltage deviation.  
- Visualize instability metrics for interpretation.  
#### Step 3: Apply Machine Learning for Stability Clustering  
- Prepare data for K-means clustering based on voltage deviation and load fluctuation.  
- Classify substations into stability categories: Stable, Moderately Unstable, and Highly Unstable.  
- Visualize the clustering results.
#### Step 4: Forecast Power Demand Using ARIMA  
- Use the auto_arima function to forecast future demand for each substation.  
- Evaluate forecast accuracy using Mean Absolute Error (MAE).  
#### Step 5: Visualize & Interpret Findings  
- Visualize power demand fluctuations over 24 hours for each substation.  
- Highlight periods of instability and analyze the results.
