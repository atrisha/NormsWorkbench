'''
Created on 11 Sept 2023

@author: Atrisha
'''

import matplotlib.pyplot as plt
import numpy as np

# Create a grid of x and y values between 0 and 1
x = np.linspace(0.5, 1, 1000)
y = np.linspace(0, 1, 1000)
X, Y = np.meshgrid(x, y)
n = 0.6
# Calculate the function values for each (x, y) pair
Z4 = n * X * Y**(1 - 0.8)
Z3 = n * X * Y**(1 - 0.6)
Z2 = n * X * Y**(1 - 0.4)
Z1 = n *  X * Y**(1 - 0.2)
mask1 = Z1 > 0.2
mask2 = Z2 > 0.2
mask3 = Z3 > 0.2
mask4 = Z4 > 0.2
# Create a 2D heatmap plot with alpha values
fig, axes = plt.subplots(2, 2, figsize=(8, 6))

axes[0,0].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
axes[0,0].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
axes[0,0].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
axes[0,0].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)

mask1 = Z1 > 0.3
mask2 = Z2 > 0.3
mask3 = Z3 > 0.3
mask4 = Z4 > 0.3


axes[0,1].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
axes[0,1].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
axes[0,1].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
axes[0,1].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)


n = 0.3
# Calculate the function values for each (x, y) pair
Z4 = n * X * Y**(1 - 0.8)
Z3 = n * X * Y**(1 - 0.6)
Z2 = n * X * Y**(1 - 0.4)
Z1 = n *  X * Y**(1 - 0.2)
mask1 = Z1 > 0.2
mask2 = Z2 > 0.2
mask3 = Z3 > 0.2
mask4 = Z4 > 0.2


axes[1,0].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
axes[1,0].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
axes[1,0].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
axes[1,0].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)

mask1 = Z1 > 0.3
mask2 = Z2 > 0.3
mask3 = Z3 > 0.3
mask4 = Z4 > 0.3


axes[1,1].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
axes[1,1].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
axes[1,1].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
axes[1,1].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)

# Add labels
plt.tight_layout()
fig.text(0.5, 0, 'Opinion degree', ha='center')
fig.text(0.04, 0.5, 'Sanctioning intensity', va='center', rotation='vertical')


# Set axis limits to [0, 1]
plt.xlim(0.5, 1)
plt.ylim(0, 1)

# Show the plot
#plt.suptitle('Expression and silence map for opinion degree and sanctioning intensity', x=0.5, y=1,)
plt.show()

