from utils import shimming_magnet
import numpy as np
#make a aoordinate grid
sim_dimension_cube_side = 300e-3 # 300 mm cube
points_per_side = 64

#%%
X = np.linspace(-sim_dimension_cube_side/2,sim_dimension_cube_side/2,points_per_side)
Y = np.linspace(-sim_dimension_cube_side/2,sim_dimension_cube_side/2,points_per_side)
Z = np.linspace(-sim_dimension_cube_side/2,sim_dimension_cube_side/2,points_per_side)

x,y,z = np.meshgrid(X,Y,Z,indexing='ij')
y2d, z2d = np.meshgrid(Y,Z,indexing='ij')


B0 = np.zeros((np.shape(x)+3))

# quick plot
from matplotlib import pyplot as plt
ax = plt.figure().add_subplot(projection='3d')
#ax.quiver(x, y, z, B0[:,:,:,0], B0[:,:,:,1], B0[:,:,:,2], length=0.1, normalize=True)    #ax.set_xlim(-0.176,0.176)
ax.plot_surface(y2d,z2d,B0[int(len(X)/2)-2,:,:,2],cmap='viridis',clim=[-10,10])
#ax.plot_surface(y2d,z2d,B0[int(len(X)/2)-2,:,:,2]-50,cmap='spring',clim=[-10,10])

ax.set_xlabel('Y')
ax.set_ylabel('Z')


    #ax.set_ylim(-0.176,0.176)
    
ax.set_title('z component of field of a single magnet YZ plane')
plt.show()