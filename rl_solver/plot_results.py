import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV files to examine their contents
agent_results_path = 'results.csv'
random_results_path = 'rand_results.csv'

agent_results = pd.read_csv(agent_results_path)
random_results = pd.read_csv(random_results_path)

agent_results.head(), random_results.head()

# Group by episode and calculate statistics for both datasets
agent_stats = agent_results.groupby('Episode')['Value'].agg(
    Median='median',
    Q25=lambda x: np.percentile(x, 25),
    Q75=lambda x: np.percentile(x, 75),
    Min='min',
    Max='max'
).reset_index()

random_stats = random_results.groupby('Episode')['Value'].agg(
    Median='median',
    Q25=lambda x: np.percentile(x, 25),
    Q75=lambda x: np.percentile(x, 75),
    Min='min',
    Max='max'
).reset_index()

# Plot the quartile river plot
plt.figure(figsize=(14, 8))

# Agent "river"
plt.fill_between(agent_stats['Episode'], agent_stats['Min'], agent_stats['Max'],
                 color='blue', alpha=0.1, label='Agent Min/Max')
plt.fill_between(agent_stats['Episode'], agent_stats['Q25'], agent_stats['Q75'],
                 color='blue', alpha=0.3, label='Agent IQR')
plt.plot(agent_stats['Episode'], agent_stats['Median'], color='blue', label='Agent Median')

# Random "river"
plt.fill_between(random_stats['Episode'], random_stats['Min'], random_stats['Max'],
                 color='orange', alpha=0.1, label='Random Min/Max')
plt.fill_between(random_stats['Episode'], random_stats['Q25'], random_stats['Q75'],
                 color='orange', alpha=0.3, label='Random IQR')
plt.plot(random_stats['Episode'], random_stats['Median'], color='orange', label='Random Median')

# Base initial value
random_stats["Initial Value"] = 8345.5
plt.plot(random_stats['Episode'], random_stats['Initial Value'], color='gray', label='Initial Eval')

# SA value
random_stats["SA Value"] = 4095
plt.plot(random_stats['Episode'], random_stats['SA Value'], color='red', label='SA Eval')


# Labels, legend, and grid
plt.title('Quartile River Plot with Min/Max: Agent vs Random Results', fontsize=16)
plt.xlabel('Episodes', fontsize=14)
plt.ylabel('Evaluations', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()