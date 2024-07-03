# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:26:51 2018

@author: to_reilly
"""
import numpy as np
import scipy.ndimage as nd

mu = 1e-7

def magnetization(bRem, dimensions, shape = 'cube', evalDistance = 1):
    #Use the analytical expression for the z component of a cube magnet to estimate
    #dipole momentstrength for correct scaling. Dipole approximation only valid 
    #far-ish away from magnet, comparison made at 1 meter distance.
    if shape == 'cube':
        b_analytic = (bRem/np.pi) *(np.arctan2(dimensions**2, 2*evalDistance *np.sqrt(4*(evalDistance**2) + 2*(dimensions**2)))-\
                        np.arctan2(dimensions**2, 2*(dimensions + evalDistance)*\
                                   np.sqrt(4*((dimensions+evalDistance)**2)+2*(dimensions**2))))
    
        dip_mom = b_analytic * (dimensions/2 + evalDistance)**3 /(2*mu) #strength of the dipole moment
    
    return dip_mom

def singleMagnet(position, dipoleMoment, simDimensions, resolution):
    #based on the dipole approximation
    #create mesh coordinates
    x = np.linspace(-simDimensions[0]/2 - position[0], simDimensions[0]/2 - position[0], int(simDimensions[0]*resolution)+1, dtype=np.float32)
    y = np.linspace(-simDimensions[1]/2 - position[1], simDimensions[1]/2 - position[1], int(simDimensions[1]*resolution)+1, dtype=np.float32)
    z = np.linspace(-simDimensions[2]/2 - position[2], simDimensions[2]/2 - position[2], int(simDimensions[2]*resolution)+1, dtype=np.float32)
    x, y, z = np.meshgrid(x,y,z)
        
    #vec_dot_dip = 3*(x*dipoleMoment[0] + y*dipoleMoment[1])
    vec_dot_dip = 3*(y*dipoleMoment[0] + z*dipoleMoment[1])
    
    #calculate the distance of each mesh point to magnet, optimised for speed
    #for improved memory performance move in to b0 calculations
    vec_mag = np.square(x) + np.square(y) + np.square(z)
    vec_mag_3 = np.power(vec_mag,1.5)
    vec_mag_5 = np.power(vec_mag,2.5)
    del vec_mag
    
    B0 = np.zeros(np.shape(x)+(3,), dtype=np.float32)

    #calculate contributions of magnet to total field, dipole always points in yz plane
    #so second term is zero for the x component
    B0[:,:,:,0] += 0#np.divide(np.multiply(x, vec_dot_dip),vec_mag_5) #- np.divide(dipoleMoment[0],vec_mag_3)
    B0[:,:,:,1] += np.divide(np.multiply(y, vec_dot_dip),vec_mag_5) - np.divide(dipoleMoment[0],vec_mag_3)
    B0[:,:,:,2] += np.divide(np.multiply(z, vec_dot_dip),vec_mag_5) - np.divide(dipoleMoment[1],vec_mag_3)
    
    return B0


def analyticMagnet(position, angle, magSize, simDimensions, resolution, bRem=1.3):
    
    #references: Engel-Herbert "Calculation of the magnetic stray field of a uniaxial magnetic domain" 

    
    #create mesh coordinates
    x = np.linspace(-simDimensions[0]/2 - position[0], simDimensions[0]/2 - position[0], int(simDimensions[0]*resolution)+1, dtype=np.float32)
    y = np.linspace(-simDimensions[1]/2 - position[1], simDimensions[1]/2 - position[1], int(simDimensions[1]*resolution)+1, dtype=np.float32)
    z = np.linspace(-simDimensions[2]/2 - position[2], simDimensions[2]/2 - position[2], int(simDimensions[2]*resolution)+1, dtype=np.float32)
    x, y, z = np.meshgrid(x,y,z)
    
    xb = magSize[0]/2
    yb = magSize[1]/2
    zb = magSize[2]/2

    bField = np.zeros(np.shape(x) + (3,))
    tempFields = np.zeros(np.shape(x) + (2,))
    
    for k in range (1,3):
        for l in range (1,3):
            for m in range (1,3):
                xTerm = x+((-1)**k)*xb
                yTerm = y+((-1)**l)*yb
                zTerm = z+((-1)**m)*zb

                #y component, direction of magnetisation
                tempFields[:,:,:,1] -= ((-1)**(k+l+m))*((yTerm*xTerm)/(np.abs(yTerm)*np.abs(xTerm)))*np.arctan((np.abs(xTerm)*zTerm)/(np.abs(yTerm)*np.sqrt(np.square(xTerm) + np.square(yTerm) + np.square(zTerm))))
                
                    
                #x component
                tempFields[:,:,:,0] += ((-1)**(k+l+m))*np.log(zTerm + np.sqrt(np.square(xTerm) + np.square(yTerm) + np.square(zTerm)))
                    
                #z component
                bField[:,:,:,2] += ((-1)**(k+l+m))*np.log(xTerm + np.sqrt(np.square(xTerm) + np.square(yTerm) + np.square(zTerm)))
    
    bField[:,:,:,0] = tempFields[:,:,:,0]*np.cos(-angle) - tempFields[:,:,:,1]*np.sin(-angle)
    bField[:,:,:,1] = tempFields[:,:,:,0]*np.sin(-angle) + tempFields[:,:,:,1]*np.cos(-angle)

    
    return (bRem/(4*np.pi))*bField

def analyticMagnetV2(position, angle, magSize, simDimensions, resolution, bRem=1.3):
    
    #references: Kazuhiro Nishimura "Three-dimensional array of strong magnetic field by using cubic permanent magnets" 

    
    #create mesh coordinates
    x = np.linspace(-simDimensions[0]/2 + position[0], simDimensions[0]/2 + position[0], int(simDimensions[0]*resolution)+1)
    y = np.linspace(-simDimensions[1]/2 + position[1], simDimensions[1]/2 + position[1], int(simDimensions[1]*resolution)+1)
    z = np.linspace(-simDimensions[2]/2 + position[2], simDimensions[2]/2 + position[2], int(simDimensions[2]*resolution)+1)
    x, y, z = np.meshgrid(x,y,z)
    # z, y, x = np.meshgrid(x,y,z)

    xb = magSize[0]/2
    yb = magSize[1]/2
    zb = magSize[2]/2

    xa = x +xb; xs = x - xb
    ya = y +yb; ys = y - yb
    za = z +zb; zs = z - zb

    bField = np.zeros(np.shape(x) + (3,))
    
    bField[...,0]  = np.log(np.abs((ya + np.sqrt(np.square(xs) + np.square(ya) + np.square(zs)))/(ys + np.sqrt(np.square(xs) + np.square(ys) + np.square(zs)))))
    bField[...,0] += np.log(np.abs((ys + np.sqrt(np.square(xa) + np.square(ys) + np.square(zs)))/(ya + np.sqrt(np.square(xa) + np.square(ya) + np.square(zs)))))
    bField[...,0] += np.log(np.abs((ys + np.sqrt(np.square(xs) + np.square(ys) + np.square(za)))/(ya + np.sqrt(np.square(xs) + np.square(ya) + np.square(za)))))
    bField[...,0] += np.log(np.abs((ya + np.sqrt(np.square(xa) + np.square(ya) + np.square(za)))/(ys + np.sqrt(np.square(xa) + np.square(ys) + np.square(za)))))
    
    
    bField[...,2]  = np.log(np.abs((xa + np.sqrt(np.square(xa) + np.square(ys) + np.square(zs)))/(xs + np.sqrt(np.square(xs) + np.square(ys) + np.square(zs)))))
    bField[...,2] += np.log(np.abs((xs + np.sqrt(np.square(xs) + np.square(ya) + np.square(zs)))/(xa + np.sqrt(np.square(xa) + np.square(ya) + np.square(zs)))))
    bField[...,2] += np.log(np.abs((xs + np.sqrt(np.square(xs) + np.square(ys) + np.square(za)))/(xa + np.sqrt(np.square(xa) + np.square(ys) + np.square(za)))))
    bField[...,2] += np.log(np.abs((xa + np.sqrt(np.square(xa) + np.square(ya) + np.square(za)))/(xs + np.sqrt(np.square(xs) + np.square(ya) + np.square(za)))))
    
    
    bField[...,1]  = (ys/np.abs(ys))*(np.arctan2(xs*np.abs(ys),(zs*np.sqrt(np.square(xs) + np.square(ys) + np.square(zs)))) - np.arctan2(xa*np.abs(ys),(zs*np.sqrt(np.square(xa) + np.square(ys) + np.square(zs)))))
    bField[...,1] -= (ya/np.abs(ya))*(np.arctan2(xs*np.abs(ya),(zs*np.sqrt(np.square(xs) + np.square(ya) + np.square(zs)))) - np.arctan2(xa*np.abs(ya),(zs*np.sqrt(np.square(xa) + np.square(ya) + np.square(zs)))))
    bField[...,1] -= (ys/np.abs(ys))*(np.arctan2(xs*np.abs(ys),(za*np.sqrt(np.square(xs) + np.square(ys) + np.square(za)))) - np.arctan2(xa*np.abs(ys),(za*np.sqrt(np.square(xa) + np.square(ys) + np.square(za)))))
    bField[...,1] += (ya/np.abs(ya))*(np.arctan2(xs*np.abs(ya),(zs*np.sqrt(np.square(xs) + np.square(ya) + np.square(za)))) - np.arctan2(xa*np.abs(ya),(zs*np.sqrt(np.square(xa) + np.square(ya) + np.square(za)))))

    
    
    return (bRem/(4*np.pi))*bField

def createHallbach(numMagnets = 24, rings = (-0.075,-0.025, 0.025, 0.075), radius = 0.145, magnetSize = 0.0254, kValue = 2, resolution = 1000, bRem = 1.3, simDimensions = (0.3, 0.3, 0.2), offsetAngle = 0, returnIndividualMagnets = False):
    
    #define vacuum permeability
    mu = 1e-7
    
    #positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2*np.pi, numMagnets, endpoint=False)

    dip_mom = magnetization(bRem, magnetSize)

    if not returnIndividualMagnets:
        #create array to store field data
        B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3), dtype=np.float32)
    else:
        totalMagnets = int(np.size(rings)*numMagnets)
        B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3,totalMagnets), dtype=np.float32)

    count = 0
    #create halbach array
    for row in rings:
        for angle in angle_elements:
            count += 1
            
            position = (radius*np.cos(angle),radius*np.sin(angle), row)
            
            dip_vec = [dip_mom*np.cos(kValue*angle + offsetAngle), dip_mom*np.sin(kValue*angle + offsetAngle)]
            dip_vec = np.multiply(dip_vec,mu)
            
            #calculate contributions of magnet to total field, dipole always points in xy plane
            #so second term is zero for the z component
            if not returnIndividualMagnets:
                B0 += singleMagnet(position, dip_vec, simDimensions, resolution)
            else:
                B0[...,count-1] = singleMagnet(position, dip_vec, simDimensions, resolution)
            
            
    simParams = {"numMagnets":numMagnets,"rings":rings,"radius":radius,"magnetSize":magnetSize,"bRemanence":bRem,"kValue":kValue,"resolution" :resolution, "SimDimensions":simDimensions,}
    
    return B0#, simParams

def createEllipticalHallbach(numMagnets = 24, rings = (-0.075,-0.025, 0.025, 0.075), semiMinor = 0.145, semiMajor = 0.145, magnetSize = 0.0254, kValue = 2, resolution = 1000, bRem = 1.3, simDimensions = (0.3, 0.3, 0.2)):
    
    #define vacuum permeability
    mu = 1e-7
    
    #positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2*np.pi, numMagnets, endpoint=False)

    dip_mom = magnetization(bRem, magnetSize)
    
    #create array to store field data
    B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3), dtype=np.float32)
    
    count = 0
    #create halbach array
    for row in rings:
        for angle in angle_elements:
            count += 1
            #print("Simulating magnet " + str(count) + " of " + str(len(rings)*len(angle_elements)))
            
            position = (semiMajor*np.cos(angle),semiMinor*np.sin(angle), row)
            
            dip_vec = [dip_mom*np.cos(kValue*angle), dip_mom*np.sin(kValue*angle)]
            dip_vec = np.multiply(dip_vec,mu)
                
            
            #calculate contributions of magnet to total field, dipole always points in xy plane
            #so second term is zero for the z component
            B0 += singleMagnet(position, dip_vec, simDimensions, resolution)
            
    # simParams = {"numMagnets":numMagnets,"rings":rings,"radius":radius,"magnetSize":magnetSize,"bRemanence":bRem,"kValue":kValue,"resolution" :resolution, "SimDimensions":simDimensions,}
    
    return B0#, simParams

def createNoisyHallbach(numMagnets = 24, rings = (-0.075,-0.025, 0.025, 0.075), radius = 0.145, magnetSize = 0.0254, kValue = 2, resolution = 1000, bRem = 1.3, simDimensions = (0.3, 0.3, 0.2), magVar = 3, angleVar = 3):
    
    #define vacuum permeability
    mu = 1e-7
    
    #positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2*np.pi, numMagnets, endpoint=False)
    
    #Use the analytical expression for the z component of a cube magnet to estimate
    #dipole momentstrength for correct scaling. Dipole approximation only valid 
    #far-ish away from magnet, comparison made at 1 meter distance.

    dip_mom = magnetization(bRem, magnetSize)
    
    #create array to store field data
    B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3), dtype=np.float32)
    
    np.random.seed()
    count = 0
    #create halbach array
    for row in rings:
        for angle in angle_elements:
            count += 1
            #print("Simulating magnet " + str(count) + " of " + str(len(rings)*len(angle_elements)))
            
            position = (radius*np.cos(angle),radius*np.sin(angle), row)
            
            angleError = np.deg2rad(np.random.uniform(-angleVar,angleVar))
            dip_vec = [dip_mom*np.cos(kValue*(angle+angleError)), dip_mom*np.sin(kValue*(angle + angleError))]
            dip_vec = np.multiply(dip_vec,mu)*(1 + np.random.uniform(-magVar/100,magVar/100))
                
            
            #calculate contributions of magnet to total field, dipole always points in xy plane
            #so second term is zero for the z component
            B0 += singleMagnet(position, dip_vec, simDimensions, resolution)
            
    simParams = {"numMagnets":numMagnets,"rings":rings,"radius":radius,"magnetSize":magnetSize,"bRemanence":bRem,"kValue":kValue,"resolution" :resolution, "SimDimensions":simDimensions,}
    
    return B0#, simParams

def createAnalyticHallbach(numMagnets = 24, rings = (-0.075,-0.025, 0.025, 0.075), radius = 0.145, magnetDims = (12*1e-3,12*1e-3,12*1e-3), kValue = 2, resolution = 1000, bRem = 1.3, simDimensions = (0.3, 0.3, 0.2), returnIndividualMagnets = False):
        
    #positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2*np.pi, numMagnets, endpoint=False)
    
    if not returnIndividualMagnets:
        #create array to store field data
        B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3), dtype=np.float32)
    else:
        totalMagnets = int(np.size(rings)*numMagnets)
        B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3,totalMagnets), dtype=np.float32)

    count = 0
    #create halbach array
    for row in rings:
        for angle in angle_elements:
            # print("Simulating magnet " + str(count) + " of " + str(len(rings)*len(angle_elements)))
            
            position = (radius*np.cos(angle),radius*np.sin(angle), row)
            
            if not returnIndividualMagnets:
                B0 += analyticMagnet(position, kValue*angle, magnetDims, simDimensions, resolution)
            else:
                B0[...,count] = analyticMagnet(position, kValue*angle, magnetDims, simDimensions, resolution)  
            count += 1

            
    # simParams = {"numMagnets":numMagnets,"rings":rings,"radius":radius,"magnetSize":magnetSize,"bRemanence":bRem,"kValue":kValue,"resolution" :resolution, "SimDimensions":simDimensions,}
    
    # return B0, simParams
    return B0

def createAnalyticHallbachV2(numMagnets = 24, rings = (-0.075,-0.025, 0.025, 0.075), radius = 0.145, magnetDims = (12*1e-3,12*1e-3,12*1e-3), kValue = 2, resolution = 1000, bRem = 1.3, simDimensions = (0.3, 0.3, 0.2), returnIndividualMagnets = False):
        
    #positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2*np.pi, numMagnets, endpoint=False)
    
    if not returnIndividualMagnets:
        #create array to store field data
        B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3), dtype=np.float32)
    else:
        totalMagnets = int(np.size(rings)*numMagnets)
        B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3,totalMagnets), dtype=np.float32)

    count = 0
    #create halbach array
    for row in rings:
        for angle in angle_elements:
            # print("Simulating magnet " + str(count) + " of " + str(len(rings)*len(angle_elements)))
            
            position = (radius*np.cos(angle),radius*np.sin(angle), row)
            
            if not returnIndividualMagnets:
                B0 += analyticMagnetV2(position, kValue*angle, magnetDims, simDimensions, resolution)
            else:
                B0[...,count] = analyticMagnetV2(position, kValue*angle, magnetDims, simDimensions, resolution)  
            count += 1

            
    # simParams = {"numMagnets":numMagnets,"rings":rings,"radius":radius,"magnetSize":magnetSize,"bRemanence":bRem,"kValue":kValue,"resolution" :resolution, "SimDimensions":simDimensions,}
    
    # return B0, simParams
    return B0


def createShimfields(numMagnets = 24, rings = 9, radius = 0.12,zRange = (-0.09,0.09), magnetRad = 0.005, magnetThick = 0.003, resolution = 1000, bRem = 1.3, kValue = 1, simDimensions = (0.1, 0.1, 0.1)):
    
    #define vacuum permeability
    mu = 1e-7
    
    #positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2*np.pi, numMagnets, endpoint=False)
    
    rings = np.linspace(zRange[0], zRange[1], rings)
    
    #Use the analytical expression for the z component of a cube magnet to estimate
    #dipole momentstrength for correct scaling. Dipole approximation only valid 
    #far-ish away from magnet, comparison made at 1 meter distance.
    magnetSize = magnetThick
    distance = 1
    b_analytic = (bRem/np.pi) *(np.arctan2(magnetSize**2, 2*distance *np.sqrt(4*(distance**2) + 2*(magnetSize**2)))-\
                        np.arctan2(magnetSize**2, 2*(magnetSize + distance)*\
                                   np.sqrt(4*((magnetSize+distance)**2)+2*(magnetSize**2))))
    dip_mom = b_analytic * (magnetSize/2 + distance)**3 /(2*mu) #strength of the dipole moment    dip_mom = b_analytic * (magnetThick/2 + distance)**3 /(2*mu) #strength of the dipole moment
    print("dipole moment of magnet: " + str(dip_mom))
    print("b_analytic: " + str(b_analytic))
    
    #create array to store field data
    shimFields = np.zeros((int(simDimensions[0]*resolution),int(simDimensions[1]*resolution),int(simDimensions[2]*resolution),3, numMagnets*np.size(rings)), dtype=np.float32)
    
    count = 0
    #create halbach array
    for row in rings:
        for angle in angle_elements:
            print("Simulating magnet " + str(count+1) + " of " + str(len(rings)*len(angle_elements)))
            
            position = (radius*np.cos(angle),radius*np.sin(angle), row)
            
            angle = kValue* angle
            dip_vec = [dip_mom*np.cos(angle), dip_mom*np.sin(angle)]
            dip_vec = np.multiply(dip_vec,mu)
                
            #create mesh coordinates
            x = np.linspace(-simDimensions[0]/2 + position[0], simDimensions[0]/2 + position[0], int(simDimensions[0]*resolution), dtype=np.float32)
            y = np.linspace(-simDimensions[1]/2 + position[1], simDimensions[1]/2 + position[1], int(simDimensions[1]*resolution), dtype=np.float32)
            z = np.linspace(-simDimensions[2]/2 + position[2], simDimensions[2]/2 + position[2], int(simDimensions[2]*resolution), dtype=np.float32)
            x, y, z = np.meshgrid(x,y,z)
                
            vec_dot_dip = 3*(x*dip_vec[0] + y*dip_vec[1])
            
            #calculate the distance of each mesh point to magnet, optimised for speed
            #for improved memory performance move in to b0 calculations
            vec_mag = np.square(x) + np.square(y) + np.square(z)
            vec_mag_3 = np.power(vec_mag,1.5)
            vec_mag_5 = np.power(vec_mag,2.5)
            del vec_mag
            
            #calculate contributions of magnet to total field, dipole always points in xy plane
            #so second term is zero for the z component
            shimFields[:,:,:,0,count] = np.divide(np.multiply(x, vec_dot_dip),vec_mag_5) - np.divide(dip_vec[0],vec_mag_3)
            shimFields[:,:,:,1,count] = np.divide(np.multiply(y, vec_dot_dip),vec_mag_5) - np.divide(dip_vec[1],vec_mag_3)
            shimFields[:,:,:,2,count] = np.divide(np.multiply(z, vec_dot_dip),vec_mag_5)
            count += 1
                
    return shimFields


def loadB0(filename):    
    try:
        B0 = np.load(filename + '.npy')
    except:
        print("Failed to load B0 data, make sure the file is correct and the file extension is not given (i.e. to load fieldData.npy give fieldData as the filename)")
        
    try:
        simParams = np.load(filename + "_params.npy").item()
        return B0, simParams
    except:
        print("Could not find a settings file")
        return B0
    return

def exportB0(B0, filename, simParams = 0):
    
    if not simParams:
        print("In order to properly be able to read in the file data next time the simulation parameters have to be included")
    else:
        np.save(filename + '_params.npy', simParams)
        
    np.save(filename + '.npy', B0)
    return
        

def rotateB0(b0_data, rotAngle):
    
    b0_data = nd.interpolation.rotate(b0_data,rotAngle,reshape=False)
    
    return b0_data

def rotateB0VectorRot(b0_data, rotAngle):
    
    b0_data = nd.interpolation.rotate(b0_data,rotAngle,reshape=False)
    b0Mag = np.sqrt(np.sum(np.multiply(b0_data,b0_data),2))
    b0_rotated = np.zeros(np.shape(b0_data))
    b0_rotated[:,:,0] = np.multiply(b0Mag,np.cos(rotAngle*np.pi/180))
    b0_rotated[:,:,1] = np.multiply(b0Mag,np.sin(rotAngle*np.pi/180))
    b0_rotated[:,:,2] = b0_data[:,:,2]
    
    return b0_rotated

def readCSTB0map(b0mapFilename, returnCoordinates = False):
    rawMap = np.genfromtxt(b0mapFilename, skip_header = 2 )

    xDims = np.unique(rawMap[:,0])
    yDims = np.unique(rawMap[:,1])
    zDims = np.unique(rawMap[:,2])
    
    b0Data = np.zeros((np.size(xDims),np.size(yDims),np.size(zDims),3))
    if returnCoordinates:
        b0Data[:,:,:,0] = np.reshape(rawMap[:,0], np.shape(b0Data[:,:,:,0]))
        b0Data[:,:,:,1] = np.reshape(rawMap[:,1], np.shape(b0Data[:,:,:,0]))
        b0Data[:,:,:,2] = np.reshape(rawMap[:,2], np.shape(b0Data[:,:,:,0]))
    else:
        b0Data[:,:,:,0] = np.reshape(rawMap[:,3], np.shape(b0Data[:,:,:,0]))
        b0Data[:,:,:,1] = np.reshape(rawMap[:,4], np.shape(b0Data[:,:,:,0]))
        b0Data[:,:,:,2] = np.reshape(rawMap[:,5], np.shape(b0Data[:,:,:,0]))

    return b0Data