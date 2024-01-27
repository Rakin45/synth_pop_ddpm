import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

data = pd.read_csv('./data/nts_toy_home_population.csv')

# Fred's encoder code
class DescreteEncoder:
    def __init__(self, duration: int = 1440, step_size: int = 10):
        self.duration = duration
        self.step_size = step_size
        self.steps = duration // step_size
        self.index_to_acts = {}
        self.acts_to_index = {}

    def encode(self, data: pd.DataFrame):
        # Create mappings from activity to index and vice versa
        self.index_to_acts = {i: a for i, a in enumerate(data.act.unique())}
        self.acts_to_index = {a: i for i, a in self.index_to_acts.items()}
        
        # Create a new DataFrame for encoded data
        encoded_data = data.copy()
        encoded_data['act'] = encoded_data['act'].map(self.acts_to_index)
        return encoded_data

encoder = DescreteEncoder()
encoded_data = encoder.encode(data)

# Function to convert encoded data into an image grid
def create_image_grid(encoded_data, encoder):
    # Map pid to sequential indices starting from 0
    pid_to_index = {pid: index for index, pid in enumerate(encoded_data['pid'].unique())}
    
    # Determine the size of the grid
    num_people = len(pid_to_index)
    time_steps = encoder.steps
    
    # Initialize an empty grid
    grid = np.zeros((num_people, time_steps))
    
    # Populate the grid with encoded activity data
    for _, row in encoded_data.iterrows():
        pid_index = pid_to_index[row['pid']]
        act_index = row['act']
        start_step = row['start'] // encoder.step_size
        end_step = row['end'] // encoder.step_size
        grid[pid_index, start_step:end_step] = act_index
    
    return grid

# Create the image grid
image_grid = create_image_grid(encoded_data, encoder)

# Check the shape of the resulting image grid and display a portion of it for visualization
image_grid.shape, image_grid[:5, :10]  # Displaying first 5 rows and 10 columns of the grid

print(image_grid.shape, image_grid[:5, :10])

plt.figure(figsize=(12, 6))
plt.imshow(image_grid, aspect='auto')
plt.colorbar()
plt.title('Activity Sequence Heatmap')
plt.xlabel('Time Steps')
plt.ylabel('Individuals')
plt.show()