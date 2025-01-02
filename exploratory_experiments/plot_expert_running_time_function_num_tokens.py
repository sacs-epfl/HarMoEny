# import pandas as pd 
# import plotly.express as px

# df = pd.read_csv("single_expert_time_vs_tokens.csv")

# fig = px.scatter(df, x="num_toks", y="time")
# fig.write_image("single_expert_time_vs_tokens.png")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Load the data
df = pd.read_csv("single_expert_time_vs_tokens.csv")

# Reshape the data for sklearn
X = df["num_toks"].values.reshape(-1, 1)  # Features (number of tokens)
y = df["time"].values  # Target (time)

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the slope (coefficient) and intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope (time per token): {slope}")
print(f"Intercept (overhead): {intercept}")

# Create a new column with the predicted values
df["predicted_time"] = model.predict(X)

# Plot the original data and the linear fit
fig = px.scatter(df, x="num_toks", y="time", title="Time vs Tokens with Linear Fit")
fig.add_scatter(x=df["num_toks"], y=df["predicted_time"], mode="lines", name="Linear Fit")
fig.write_image("single_expert_time_vs_tokens.png")
