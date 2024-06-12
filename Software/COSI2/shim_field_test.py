# tests of the shimming fields.
# define area, -1,1 around 0, 3d, make a meshgrid.
# make a magnet. position center, moment in yz along y,
# no rotation.
# pass the meshgrid to the magnet, calculate the field.
import numpy as np
from matplotlib import pyplot as plt
from utils import shimming_magnet



# make a 3d meshgrid
X = np.linspace(-1,1,100)
Y = np.linspace(-1,1,200)
Z = np.linspace(-1,1,300)

x,y,z = np.meshgrid(X,Y,Z, indexing='ij')

# temp: make field of one magnet
# todo: move this into the magnet class, mind the magnitude of the moment
dipoleMoment = [0,1,0] # originally || oy
mx = dipoleMoment[0]
my = dipoleMoment[1]
mz = dipoleMoment[2]
vec_dot_dip = 3*(np.multiply(mx,x) + np.multiply(my,y) + np.multiply(mz,z))
rvec = np.sqrt(np.square(x)+np.square(y)+np.square(z))    

B0 = np.zeros(np.shape(x)+(3,), dtype=np.float32)
B0[:,:,:,0] = np.divide(np.multiply(x,vec_dot_dip),rvec**5)# - np.divide(mx,rvec**3)
B0[:,:,:,1] = np.divide(np.multiply(y,vec_dot_dip),rvec**5) - np.divide(my,rvec**3)
B0[:,:,:,2] = np.divide(np.multiply(z,vec_dot_dip),rvec**5) - np.divide(mz,rvec**3)

axB = plt.figure().add_subplot()
axB.imshow(B0[int(len(X)/1.2),:,:,2])


# make an array of magnets

alpha_pos = np.linspace(0,2*np.pi,128)
x_pos = np.zeros(len(alpha_pos))
ring_radius = 0.3 # [m]
y_pos = ring_radius*np.cos(alpha_pos)
z_pos = ring_radius*np.sin(alpha_pos)
my_magnets = []

for i in range(len(x_pos)):
    my_magnet = shimming_magnet.shimming_magnet(position=[x_pos[i],y_pos[i],z_pos[i]], dipole_moment = 0.1, rotation_yz = np.pi/17*i)
    my_magnets.append(my_magnet)

# plot the magnet positions

ax = plt.figure().add_subplot(projection='3d')
#ax.quiver(x, y, z, B0[:,:,:,0], B0[:,:,:,1], B0[:,:,:,2], length=0.1, normalize=True)    #ax.set_xlim(-0.176,0.176)
for magnet in my_magnets:
    #ax.plot(xs=[magnet.position[0],magnet.position[0]+magnet.dipole[0]],ys=[magnet.position[1],magnet.position[1]+magnet.dipole[1]],zs=[magnet.position[2],magnet.position[2]+magnet.dipole[2]],color='k')
    ax.quiver(magnet.position[0],magnet.position[1],magnet.position[2],magnet.dipole[0],magnet.dipole[1],magnet.dipole[2])
plt.show()
