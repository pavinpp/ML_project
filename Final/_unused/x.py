import matplotlib.pyplot as plt
import numpy as np

# --- Parameters ---
start_year = 2025
stop_injection_year = 2055
end_plot_year = 2060

n_wells = 3
target_per_well_mmt = 20  # 20 Million Metric Tonnes per well
total_target_mmt = n_wells * target_per_well_mmt # 60 MMT
goal_mmt = 50 # The specific goal line requested

# Convert MMT to Tonnes for the Y-axis (Standard Condition)
target_per_well_tonnes = target_per_well_mmt * 1e6
total_target_tonnes = total_target_mmt * 1e6
goal_tonnes = goal_mmt * 1e6

# --- Time Array ---
# Create an array of years (including partial steps for smoother lines if needed, 
# but yearly integers are usually fine for this scale)
years = np.arange(start_year, end_plot_year + 1, 0.1)

# --- Calculation Functions ---

def get_cumulative_mass(current_year, max_capacity, start, stop):
    """Calculates cumulative mass at a specific time t."""
    if current_year < start:
        return 0.0
    elif current_year <= stop:
        # Linear injection rate
        duration = stop - start
        rate = max_capacity / duration
        return rate * (current_year - start)
    else:
        # Injection stopped, mass stays constant
        return max_capacity

# Vectorize function to apply to numpy array
v_get_cumulative = np.vectorize(get_cumulative_mass)

# Calculate data for one well
well_data = v_get_cumulative(years, target_per_well_tonnes, start_year, stop_injection_year)

# Calculate total data (sum of 3 wells)
total_data = well_data * n_wells

# --- Plotting ---
plt.figure(figsize=(12, 6))

# 1. Plot Individual Wells (Dashed styles similar to sample)
# Well 1 (Cyan-ish)
plt.plot(years, well_data, color='cyan', linestyle='--', linewidth=1.5, label='Well_1_Cumulative')
# Well 2 (Green-ish) - Offset slightly just for visibility if they are identical, 
# typically in sims they overlap perfectly if parameters are identical. 
# I will plot them identical as requested by logic, but you can see them in the legend.
plt.plot(years, well_data, color='lime', linestyle='--', linewidth=1.5, dashes=(5, 5), label='Well_2_Cumulative')
# Well 3 (Red-ish)
plt.plot(years, well_data, color='red', linestyle='-.', linewidth=1.5, label='Well_3_Cumulative')

# 2. Plot Total Sum (Thick Dark Red/Brown Line)
plt.plot(years, total_data, color='#8B0000', linewidth=2.5, label='Total_Injection_Mass')

# 3. Plot Goal Line (50 MMT)
plt.axhline(y=goal_tonnes, color='black', linestyle=':', linewidth=2, label=f'Goal ({goal_mmt} MMT)')

# --- Formatting to match the engineering style ---

# Y-Axis Formatting (Scientific notation)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.ylabel('Cumulative Gas Mass SC (tonne)', fontweight='bold')

# X-Axis Formatting
plt.xlabel('Date', fontweight='bold')
plt.xlim(start_year, end_plot_year)
plt.xticks(np.arange(start_year, end_plot_year + 1, 2)) # Ticks every 2 years

# Grid
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.minorticks_on()
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

# Title
plt.title('Cumulative Gas Mass SC - Injection Simulation (2025-2055)', fontweight='bold')

# Legend
plt.legend(loc='upper left', framealpha=0.9)

# Adjust layout and save/show
plt.tight_layout()

# Save logic is handled by the environment, here we just show
plt.show()