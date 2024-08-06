from cryojax.image import map_coordinates, map_coordinates_with_cubic_spline

import numpy as np
## plotting settings
from matplotlib import pyplot as plt

def test_map_coordinates():
    print(map_coordinates, map_coordinates_with_cubic_spline)
    
    ## Create a single Gaussian centered in the middle
    sigma = 0.2
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    xyz = np.meshgrid(x, y, z, indexing='ij')
    r = np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2)
    data = np.exp(- r**2 / (2 * sigma**2))
    print(data.shape)
    
    ## Plot the center slice
    plt.imshow(data[:, :, 50])
    plt.savefig("./plots/test_map_coordinates_0.png")
    
    x_slice = np.linspace(-1, 1, 100)
    y_slice = np.linspace(-1, 1, 100)
    
    
if __name__ == "__main__":
    test_map_coordinates()
    

