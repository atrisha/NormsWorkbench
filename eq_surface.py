from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt


# Parameters
  # Example parameters for h, o, a

def surface_difference_updated(point, h, o, a):
    x, y, z = point
    surface1 = (h * o * (1 - x) / (2 * a + y - (1 - h) * o))**(1 / x) if x != 0 else 0
    surface2 = (h * o * (1 - x) / (2 * a + y - (1 - h) * o))**(1 / x) if x != 0 else 0
    epsilon = 1e-6
    x = x if x != 0 else epsilon
    y = y if y != 0 else epsilon

    term1 = (h**2 * (1 - z)) / (2*a + y - (1 - h)*h)
    term2 = ((1 - h)**2 * (1 - y)) / (2*a + x - (1 - h)*h)

    # The new formula for the surface
    surface2 = 0.5 * (term1**(1/z)) 
    surface3 = 0.5 * (term2**(1/y))
    # The function should return a vector of differences for each dimension
    return [surface1, surface2, surface3]
_x,_y=[],[]
for o in np.linspace(0.5,1,100):
    h, a = 0.8, 0.09
    # Find intersection points with the updated function
    _pts = np.linspace(0.1,0.9,100)
    initial_guesses = [(x,x,x) for x in _pts]
    intersection_points_updated = []
    for guess in initial_guesses:
        result, _, _, _ = fsolve(surface_difference_updated, guess, args=(h, o, a), full_output=True)
        intersection_points_updated.append(result)
    
    intersection_points_updated
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    '''
    # Convert the intersection points to arrays for plotting
    intersection_points_array = np.array(intersection_points_updated)
    p = np.mean(intersection_points_array,axis=0)
    print(p)
    _x.append(o)
    _y.append(p[2])
plt.plot(_x,_y,'o')
plt.show()
'''
# Plot the 3D line through the intersection points
ax.plot(intersection_points_array[:, 0], intersection_points_array[:, 1], intersection_points_array[:, 2],
        color='green', marker='o', label='Intersection Line')

# Set labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title("3D Line Through Intersection Points")
plt.legend()

plt.show()
'''