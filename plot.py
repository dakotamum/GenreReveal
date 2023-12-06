import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv('tracks_output.csv')

# get names of columns
col1_name = data.columns[0]
col2_name = data.columns[1]
col3_name = data.columns[2]

# get column data
col1 = data[col1_name]
col2 = data[col2_name]
col3 = data[col3_name]
c = data['c']  # Assuming 'c' is the column for color data

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(col1, col2, col3, c=c, cmap='viridis')

# axis labels
ax.set_xlabel(col1_name)
ax.set_ylabel(col2_name)
ax.set_zlabel(col3_name)

# Save the plot as a jpg
plt.savefig('3d_plot.jpg')

# display the plot
plt.show()
