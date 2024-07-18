import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the data and the header
data = pd.read_csv("morse_log.txt", sep="\t")
header = list(data.columns)

# Identify the concentration and epsilon columns
concentration_cols = [col for col in header if "conc" in col]
epsilon_cols = [col for col in header if "Eps" in col]
target_yield_cols = [col for col in header if "Target_Yield" in col]

# Define the colormap and shuffle
colors = plt.cm.tab20(np.linspace(0, 1, 20))
np.random.shuffle(colors)

# Ensure we have enough colors
if len(colors) < len(concentration_cols) + len(epsilon_cols) + len(target_yield_cols):
    additional_colors_needed = len(concentration_cols) + len(epsilon_cols) + len(target_yield_cols) - len(colors)
    extra_colors = plt.cm.tab20(np.linspace(0, 1, additional_colors_needed))
    colors = np.concatenate((colors, extra_colors))
np.random.shuffle(colors)

# Create the figure and subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

# Plotting the concentration data
for i, conc_col in enumerate(concentration_cols):
    ax1.plot(data["Iteration"], data[conc_col], label=conc_col, color=colors[i])

ax1.set_xlabel("Iterations")
ax1.set_ylabel("Concentrations")
ax1.set_title("Concentrations vs Iterations")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(alpha=0.3)

# Plotting the epsilon data
for i, eps_col in enumerate(epsilon_cols):
    ax2.plot(data["Iteration"], data[eps_col], label=eps_col, color=colors[len(concentration_cols) + i])

ax2.set_xlabel("Iterations")
ax2.set_ylabel("Epsilons")
ax2.set_title("Epsilons vs Iterations")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(alpha=0.3)

# Plotting the target yield data
for i, yield_col in enumerate(target_yield_cols):
    ax3.plot(data["Iteration"], data[yield_col], label=yield_col, color=colors[len(concentration_cols) + len(epsilon_cols) + i])

ax3.set_xlabel("Iterations")
ax3.set_ylabel("Yield")
ax3.set_title("Yield vs Iterations")
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(alpha=0.3)

# Adjust layout to make room for the legends
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save the plot
plt.savefig("optimization_morse.pdf")








