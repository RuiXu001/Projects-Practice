# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(R_Value, F_Value)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''

plt.plot(dataset)
plt.title(dataset.columns[0])
plt.show()
'''
x = dataset['R_Value']
y = dataset['F_Value']
z = dataset['M_Value']

r = dataset['R_score'].astype('int')/5 + 0.1
g = dataset['F_score'].astype('int')/5 + 0.1
b = dataset['M_score'].astype('int')/5 + 0.1
colors = [[i,j,k] for i,j,k in zip(r,g,b)]

# Creating figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax = ax = Axes3D(fig)

# Creating plot
ax.scatter(x, y, z, color = colors)
plt.title("RFM 3D scatter plot")
ax.set_xlabel('R value', fontweight ='bold')
ax.set_ylabel('F value', fontweight ='bold')
ax.set_zlabel('M value', fontweight ='bold')
# show plot
plt.show()


'''Syntax: view_init(elev, azim)
Parameters: 
‘elev’ stores the elevation angle in the z plane.
‘azim’ stores the azimuth angle in the x,y plane.D constructor.'''
