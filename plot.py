import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv('tracks_output.csv')

col1 = data["danceability"]
col2 = data["loudness"]
col3 = data["instrumentalness"]
c = data['c']

# create 3d plot
print("checkpoint1")
fig = plt.figure()
print("checkpoint2")
ax = fig.add_subplot(111, projection='3d')
print("checkpoint3")

scatter = ax.scatter(col1, col2, col3, c=c, cmap='viridis')

# axis labels
ax.set_xlabel(col1.name)
ax.set_ylabel(col2.name)
ax.set_zlabel(col3.name)

# save the plot as a jpg
plt.savefig('3d_plot.jpg')

# # display the plot
plt.show()
