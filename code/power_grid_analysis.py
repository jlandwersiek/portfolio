import numpy as np  # numerical operations for power demand simulation
import pandas as pd  # handling time-series demand data
import networkx as nx  # modeling the power grid as a graph
import matplotlib.pyplot as plt  # visualization of demand and grid structure
from sklearn.cluster import KMeans  # machine learning for stability clustering
from pmdarima import auto_arima  # auto_arima for automatic ARIMA model selection
from sklearn.metrics import mean_absolute_error  # for evaluating forecast accuracy

# set a random seed for reproducibility in simulations
np.random.seed(42)

# step 1: define the power grid structure
grid = nx.Graph()
substations = ["Sub_A", "Sub_B", "Sub_C", "Sub_D", "Sub_E", "Sub_F", "Sub_G", "Sub_H"]
grid.add_nodes_from(substations)

connections = [
    ("Sub_A", "Sub_B"), ("Sub_B", "Sub_C"), ("Sub_C", "Sub_D"),
    ("Sub_D", "Sub_E"), ("Sub_E", "Sub_F"), ("Sub_F", "Sub_G"),
    ("Sub_G", "Sub_H"), ("Sub_H", "Sub_A"),
    ("Sub_B", "Sub_F"), ("Sub_C", "Sub_G")
]
grid.add_edges_from(connections)

# Step 1: Visualize the Power Grid Structure
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(grid)  # positions for all nodes
nx.draw_networkx_nodes(grid, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(grid, pos, width=2, alpha=0.7, edge_color='gray')
nx.draw_networkx_labels(grid, pos, font_size=12, font_weight='bold', font_color='black')
plt.title("Power Grid Structure")
plt.axis("off")  # turn off the axis
plt.show()

# step 2: generate simulated power demand data
time_steps = 24  # 24 hours for simulation
demand_data = {}
for substation in substations:
    base_load = np.random.randint(50, 100)  # base load (mw) for each substation
    fluctuation = np.random.normal(0, 10, time_steps)  # simulated demand fluctuation
    demand = np.clip(base_load + fluctuation, 30, 120)  # enforce demand limits (30-120 mw)
    demand_data[substation] = demand

df_demand = pd.DataFrame(demand_data)

# step 3: analyze stability using statistical methods
# 1. voltage deviation (simulated as the variation in demand)
voltage_deviations = df_demand.std(axis=0)  # standard deviation of demand for each substation

# 2. load fluctuations (difference in demand between consecutive hours)
load_fluctuations = df_demand.diff().abs().mean(axis=0)  # mean of absolute hourly differences

# 3. detecting instability (using a basic threshold for large deviations)
instability_threshold = 20  # example threshold for instability (in mw)
unstable_substations = voltage_deviations[voltage_deviations > instability_threshold]

# check if any substations are unstable
if unstable_substations.empty:
    print("no substations are unstable (voltage deviation below threshold)")
else:
    print("unstable substations (demand deviation > 20 mw):")
    print(unstable_substations)

# print the results
print("\nvoltage deviations (mw):")
print(voltage_deviations)
print("\nload fluctuations (mw):")
print(load_fluctuations)

# forecasting: use auto_arima to predict future demand for each substation
forecast_steps = 6  # forecast for the next 6 hours

forecast_data = {}
for substation in substations:
    # Fit ARIMA model to the demand data using auto_arima
    model = auto_arima(df_demand[substation], seasonal=False, stepwise=True, trace=True)
    forecast = model.predict(n_periods=forecast_steps)
    forecast_data[substation] = forecast

# Convert forecasted data into a DataFrame
df_forecast = pd.DataFrame(forecast_data)

# print forecast results
print("\nForecasted power demand for the next 6 hours (MW):")
print(df_forecast)

# Evaluate forecast accuracy using Mean Absolute Error (MAE)
mae_scores = {}
for substation in substations:
    # Compare forecasted values with the actual demand for the last 6 hours
    actual_values = df_demand[substation][-forecast_steps:].values
    forecast_values = df_forecast[substation].values
    mae = mean_absolute_error(actual_values, forecast_values)
    mae_scores[substation] = mae

# print MAE scores for each substation
print("\nMean Absolute Error for forecast (MW):")
print(mae_scores)

# step 4: apply machine learning for stability clustering
# prepare data for clustering
stability_features = pd.DataFrame({
    "voltage_deviation": voltage_deviations,
    "load_fluctuation": load_fluctuations
})

# apply k-means clustering to categorize stability levels
kmeans = KMeans(n_clusters=3, random_state=42)
stability_features["cluster"] = kmeans.fit_predict(stability_features)

# label the clusters
stability_labels = {0: "stable", 1: "moderately unstable", 2: "highly unstable"}
stability_features["stability_category"] = stability_features["cluster"].map(stability_labels)

# print clustering results
print("\nstability classification:")
print(stability_features[["voltage_deviation", "load_fluctuation", "stability_category"]])

# visualize clustering results
plt.figure(figsize=(10, 6))
colors = {"stable": "green", "moderately unstable": "yellow", "highly unstable": "red"}
for category, color in colors.items():
    subset = stability_features[stability_features["stability_category"] == category]
    plt.scatter(subset["voltage_deviation"], subset["load_fluctuation"], label=category, color=color)

plt.xlabel("voltage deviation (mw)")
plt.ylabel("load fluctuation (mw)")
plt.title("stability clustering of substations")
plt.legend()
plt.show()

# step 5: visualize & interpret findings
plt.figure(figsize=(12, 6))
for substation in substations:
    plt.plot(df_demand.index, df_demand[substation], label=substation)
plt.title("power demand fluctuations over 24 hours")
plt.xlabel("hour")
plt.ylabel("power demand (mw)")
plt.legend()
plt.show()

# highlight instability periods
plt.figure(figsize=(10, 6))
plt.bar(stability_features.index, stability_features["voltage_deviation"], color=[colors[cat] for cat in stability_features["stability_category"]])
plt.xlabel("substation")
plt.ylabel("voltage deviation (mw)")
plt.title("stability classification of substations")
plt.show()
