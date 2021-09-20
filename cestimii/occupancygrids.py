# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:07:47 2021

@author: Simon Biberger

Cestimii.
Compute curvature estimations on point clouds using integral invariants.

This file contains all methods that have something to do with the 
initialization, construction, splitting or merging of occupancy grids or
occupancy grids of boundary boxes. Both on strict and relaxed occupancy grids.
"""

import time
import warnings
import numpy as np
import scipy.ndimage as nd

#these three imports are only necessary for imgstack method
from PIL import Image
import h5py
import os

import cestimii.geomshapes as gs

def constructoccgrid_sphere(r=6, samp=0):
    """
    Constructs a (strict) occupancy grids of a spheres of radius 'r'.
    
    Input is a parameter 'r', which specifies the radius of the sphere. If r
    is negative, an alternative, more accurate, but less efficient way to
    generate a sphere of radius -r is used.
    
    While the "samp" parameter, which specifies the amount of points used to
    create the sphere's point cloud, often yields good results for 360*5 for
    a wide variety of scales, setting samp=0, uses a formula to dynamically
    adapt the sample rate to the radius to ensure the sphere's occupancy grid
    is always initialized correctly.
    
    Returns a matrix 'OGS', which represent the Occupancy Grid
    of the Sphere with radius 'r'. OGB has dimensions (2*r+1)x(2*r+1)x(2*r+1).
    
    The default sphere radius 'r' is 6.
    Formerly known as "createsphereoccgrid()".
    """
    if samp==0:
        samp=360*r/2
    
    if (r == 0.5) | (r == 1.5) | (r < 0 ):
        if r == 0.5:
            OGB = np.ones((1,1,1), int)
            
            return OGB
        if r == 1.5:
            OGB = np.ones((3,3,3), int)
            OGB[0,0,0] = 0
            OGB[0,0,2] = 0
            OGB[0,2,0] = 0
            OGB[0,2,2] = 0
            OGB[2,0,0] = 0
            OGB[2,0,2] = 0
            OGB[2,2,0] = 0
            OGB[2,2,2] = 0
            OGB[1,1,1] = 0
            
            return OGB
        if r < 1: #this kernel might be more precise
            radius = -r
            W = 2*radius +1
            center=(radius, radius, radius)
            a=[]
            b=[]
            c=[]
            OGB=np.zeros((W,W,W), np.int8)
            for z in range(W):
                for y in range(W):
                    for x in range(W):
                        val=(x-center[2])**2+(y-center[1])**2+(z-center[0])**2
                        if val==radius**2:
                            a.append(x)
                            b.append(y)
                            c.append(z)
                            OGB[z,y,x]=1
                            
            return OGB
    elif type(r)!=int:
        raise NameError('Radius has to be a integer.')
    else:
        
        #load sphere pointcloud
        x,y,z = gs.sphere(r=100, c=[0,0,0], 
                           samp=samp)
        
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        
        #find min,max values of each axis
        xmin =np.min(x)
        xmax =np.max(x)
        ymin =np.min(y)
        ymax =np.max(y)
        zmin =np.min(z)
        zmax =np.max(z)
        
        #normalise the axes, i.e. rescale the object (!)
        xn = ((x-xmin)/(xmax-xmin))
        yn = ((y-ymin)/(ymax-ymin))
        zn = ((z-zmin)/(zmax-zmin))
        
        #init grid over whole ball domain
        OGS = np.zeros([2*r+1,2*r+1,2*r+1], int) #see sketch on rmkbl tablet
        
        #populate ballgrid boundary
        OGS[(np.round(xn * 2*r)).astype(int),
            (np.round(yn * 2*r)).astype(int),
            (np.round(zn * 2*r)).astype(int)] = 1
        
        return OGS
       
def constructoccgrid_ball(r=6, samp=0):
    """
    Constructs a (strict) occupancy grid of a ball of radius 'r'.
    
    Input is a parameter 'r', which specifies the radius of the ball. If r
    is negative, an alternative, more accurate, but less efficient way to
    generate a sphere of radius -r is used. As the kernel is quite inaccurate
    for small values of r, I hardcoded two small kernels for r=1 and r=1.5,
    whereby the latter should serve as a better kernel for r=2.
    
    While the "samp" parameter, which specifies the amount of points used to
    create the sphere's point cloud, often yields good results for 360*5 for
    a wide variety of scales, setting samp=0, uses a formula to dynamically
    adapt the sample rate to the radius to ensure the sphere's occupancy grid
    is always initialized correctly.
    
    Returns a matrix 'OGB', which represent the Occupancy Grid
    of the Ball with radius 'r'. OGB has dimensions (2*r+1)x(2*r+1)x(2*r+1).
    
    The default ball radius 'r' is 6.
    Formerly known as "createballoccgrid()".
    """
    if samp==0:
        samp=360*r//2
    
    if (r == 0.5) | (r == 1.5) | (r < 0 ):
        if r == 0.5:
            OGB = np.ones((1,1,1), int)
            
            return OGB
        if r == 1.5:
            OGB = np.ones((3,3,3), int)
            OGB[0,0,0] = 0
            OGB[0,0,2] = 0
            OGB[0,2,0] = 0
            OGB[0,2,2] = 0
            OGB[2,0,0] = 0
            OGB[2,0,2] = 0
            OGB[2,2,0] = 0
            OGB[2,2,2] = 0
            
            return OGB
        if r < 1: #more accurate, but less efficient.
            radius = -r
            W = 2*radius +1
            center=(radius, radius, radius)
            a=[]
            b=[]
            c=[]
            OGB=np.zeros((W,W,W), np.int8)
            for z in range(W):
                for y in range(W):
                    for x in range(W):
                        val=(x-center[2])**2+(y-center[1])**2+(z-center[0])**2
                        if val<=radius**2:
                            a.append(x)
                            b.append(y)
                            c.append(z)
                            OGB[z,y,x]=1
                            
            return OGB
    elif type(r)!=int:
        raise NameError('Radius has to be an integer.')
    else:
        #load pointcloud
        x,y,z = gs.sphere(r=100, c=[0,0,0], 
                           samp=samp)
        
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        
        #find min,max values of each axis
        xmin =np.min(x)
        xmax =np.max(x)
        ymin =np.min(y)
        ymax =np.max(y)
        zmin =np.min(z)
        zmax =np.max(z)
        
        #normalise the axes, i.e. rescale the object (!)
        xn = ((x-xmin)/(xmax-xmin))
        yn = ((y-ymin)/(ymax-ymin))
        zn = ((z-zmin)/(zmax-zmin))
        
        #init grid over whole ball domain
        OGB = np.zeros([2*r+1,2*r+1,2*r+1], int)
        
        #populate ballgrid boundary
        OGB[(np.round(xn * 2*r)).astype(int),
            (np.round(yn * 2*r)).astype(int),
            (np.round(zn * 2*r)).astype(int)] = 1
        
        #fill ballgrid
        OGB = nd.binary_fill_holes(OGB).astype(int)
        
        return OGB
    
def constructoccgrid_ballzone(kr, alpha):
    """
    Constructs an (strict) occupancy grid of a ballzone of with 'alpha', i.e. 
    difference of a ball of radius 'r' and a ball of radius 'r-alpha'.
    
    Input:
        -'r': specifies the radius of the ball
        -'alpha': is used for the inner circle of the zonal ball area, i.e.
                  the inner circle is of radius r-alpha
        -'r', which specifies the radius of the ball. If r
         is negative, an alternative, more accurate, but less efficient way to
         generate a sphere of radius -r is used. As the kernel is quite 
         inaccurate for small values of r, I hardcoded two small kernels for 
         r=1 and r=1.5, whereby the latter should serve as a better kernel for 
         r=2.
    
    Returns a matrix 'OGBZ', which represent the Occupancy Grid
    of the ballzone. I.e. we return the difference between the ball kernel of
    size r and the ball kernel of size r-alpha. This ballzone is alpha wide.
    """
    #get occgrid of ball with radius r
    rball = constructoccgrid_ball(r=kr) #dims (2xkr+1)**3
    
    #get occgrid of ball with radius r
    ralphaball = constructoccgrid_ball(r=kr-alpha) #(2x(kr-alpha) +1)**3
    
    #pad occgrid ralphaball to the size of the occgrid rball
    ralphaball = np.pad(ralphaball,((alpha, alpha),(alpha, alpha),
                                    (alpha, alpha)), 'constant')
    
    OGBZ = rball-ralphaball
    
    return OGBZ

def constructoccgrid_pointcloud(inp, rho, ocg="str", taulow=0, fill=1):
    """
    Construct a (strict or relaxed) occupancy grid of arbitrary pointclouds.
    
    Input:
        -'inp': inp can be a .xyz pointcloud in the format x y z, OR a list or
                array in the format of z y x.
        -'rho': controls the amount of cells in the occupancy grid.
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the 
         usage of a relaxed occupancy grid.
        -'fill': controls wether occupancy grid will be filled or not. 
                 'fill'==1 means it is filled (using 
                 ndimage.binary_fill_holes() ).
    
    Due to the resizing with rho, the output occupancy grid might be of 
    lower resolution than the actual dataset.
    
    If the occupancy grid is not filled correctly, try reducing 'rho' or use
    a point cloud that is sampled more densely. A relaxed occgrid will not
    be filled.
    
    Returns a matrix 'OGD', which represents the Occupancy Grid of the 
    pointcloud over the whole volume Domain.
    """
    if type(inp)==str:
        #load pointcloud
        if ocg=="str":
            x, y, z = np.loadtxt(inp,skiprows=1, unpack=True,
                             usecols=(0,1,2))
        else:
            x, y, z, vals = np.loadtxt(inp,skiprows=1, unpack=True,
                             usecols=(0,1,2,3))
    elif isinstance(inp,np.ndarray):
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]    
        
        if ocg=="rlx":
            vals = inp[:,3]
    elif isinstance(inp,list):
        #we assume the input list is given like [z,y,x]
        inp = np.transpose(inp) # shape is now [#pts, 3] (= z y x)

        #get the separate coordinate vectors
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]
        
        if ocg=="rlx":
            vals = inp[:,3]
    else:
        raise NameError('Input can be an already loaded pointcloud that \
                        consists of the three coordinates z y x or a string \
                        that leads to the file that is in z y x format with \
                        no header.')
    
    if ocg=="rlx":
        #first, shift values if they are negative, then normalize vals s.t. 
        #max is 1, then multiply all values such that the maximum is sc
        valsmin = np.min(vals)
        if valsmin<0:
            vals = vals-valsmin #shift to 0
        valsmax = np.max(vals)
        if valsmax==0:
            raise NameError('Maximum Value of the point cloud is 0. Cant \
                            divide by zero')
        vals = (vals/valsmax)
    
    #find min,max values of each axis
    xmin =np.min(x)
    ymin =np.min(y)
    zmin =np.min(z)
    
    #normalise x and then rescale axes to max val
    # xrs = ((x-xmin)/(xmax-xmin))*xmax
    # yrs = ((y-ymin)/(ymax-ymin))*ymax
    # zrs = ((z-zmin)/(zmax-zmin))*zmax
    xrs = (x-xmin)
    yrs = (y-ymin)
    zrs = (z-zmin)
    
    xmax = np.max(xrs)
    ymax = np.max(yrs)
    zmax = np.max(zrs)
    
    if ocg=="str":
        #init grid over domain
        OGD = np.zeros([int(zmax*rho)+1,int(ymax*rho)+1,int(xmax*rho)+1], bool)
        
        #populate domaingrid boundary
        OGD[(zrs * rho).astype(int),
            (yrs * rho).astype(int),
            (xrs * rho).astype(int)] = 1
        
        
        #fill domaingrid
        if fill==1:
            OGD = nd.binary_fill_holes(OGD).astype(int)
    else:
        #init grid over domain
        OGD = np.zeros([int(zmax*rho)+1,int(ymax*rho)+1,int(xmax*rho)+1], 
                       np.double)
        
        #populate domaingrid boundary but only for values >= taulow
        okvals = vals >= taulow
        okindc = np.where(okvals==1)
        
        OGD[(zrs[okindc[0]] * rho).astype(int),
            (yrs[okindc[0]] * rho).astype(int),
            (xrs[okindc[0]] * rho).astype(int)] = vals
    
    return OGD

def constructpcagrids_pointcloud(inp, rho, kr=6, ocg="str", taulow=0, 
                                 variant=1, debug=0):
    """
    Construct all necessary occupancy grids for the calculation of the
    curvature estimation using integral invariants and PCA.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         coordinates x y z or a string that leads to the file that is in x y z
         format with no header. 
        -'rho' controls the amount of cells in the occupancy grid (=rho+1).
        -'kr' is the kernel radius
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the 
         usage of a relaxed occupancy grid.
        -'variant': there are two variants to calculate these values, which
                    are equivalent, but return two different outputs.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Returns the Occupancy Grid of the Domain 'OGD', the Occupancy Grid of the
    Ball neighborhoord 'OGB', and the other two Occupancy Grids necessary for
    the convolution, which are OGB*(-x), here 'OGBX', and OGB*(x*x^T), here
    called 'OGBXX'.
    """
    if debug==1:
        starttime = time.time()    
    
    if type(inp)==str:
        #load pointcloud
        if ocg=="str":
            x, y, z = np.loadtxt(inp,skiprows=1, unpack=True,
                             usecols=(0,1,2))
        else:
            x, y, z, vals = np.loadtxt(inp,skiprows=1, unpack=True,
                             usecols=(0,1,2,3))
    elif isinstance(inp,np.ndarray):
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]
        
        if ocg=="rlx":
            vals = inp[:,3]
    elif isinstance(inp,list):
        #we assume the input list is given like [z,y,x]
        inp = np.transpose(inp) # shape is now [#pts, 3] (= z y x)

        #get the separate coordinate vectors
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]
        
        if ocg=="rlx":
            vals = inp[:,3]
    else:
        raise NameError('Input can be an already loaded pointcloud that \
                        consists of the three coordinates z y x or a string \
                        that leads to the file that is in z y x format with \
                        no header.')
                        
    if debug==1:
        print("Initialised the input point cloud.\n"
              + "Current Runtime: " + str(time.time() - starttime))
        print("Number of Points in the pointcloud: " + str(np.shape(x)[0]))
    
    if ocg=="str":
        OGD = constructoccgrid_pointcloud([z,y,x], rho)
    else:
        OGD = constructoccgrid_pointcloud([z,y,x, vals], rho, ocg=ocg, 
                                          taulow=taulow, fill=0)
    
    OGB = constructoccgrid_ball(kr)
    
    if (np.shape(OGB)[0] %2 == 0 | 
        np.shape(OGB)[1] %2 == 0 | 
        np.shape(OGB)[2] %2 == 0):
        warnings.warn("The lengths of the kernel should be uneven s.t. there\
                      is a proper center element.")
                      
    if variant==0:
        ogbx, ogby, ogbz, ogbxx, ogbyy, ogbzz, ogbxy,\
        ogbxz, ogbyz = np.zeros([9, np.shape(OGB)[0], np.shape(OGB)[1], 
                                 np.shape(OGB)[2]])
        
        cx=np.shape(OGB)[1] // 2
        cy=np.shape(OGB)[0] // 2
        cz=np.shape(OGB)[2] // 2
        coords = np.ones([np.shape(OGB)[0],np.shape(OGB)[1],np.shape(OGB)[2]])
        
        xcoord = np.linspace(-cx,np.shape(OGB)[1]-1-cx,np.shape(OGB)[1])
        xcoords = np.multiply(np.transpose(coords,(0,2,1)),xcoord)
        ogbx = np.multiply(OGB, np.transpose(-xcoords,(0,2,1)))
        ogbxx = np.multiply(OGB, np.transpose(xcoords**2,(0,2,1)))
        
        ycoord = np.linspace(-cy,np.shape(OGB)[0]-1-cy,np.shape(OGB)[0])
        ycoords = np.multiply(np.transpose(coords,(2,1,0)),ycoord)
        ogby = np.multiply(OGB, np.transpose(-ycoords,(2,1,0)))
        ogbyy = np.multiply(OGB, np.transpose(ycoords**2,(2,1,0)))
        
        zcoord = np.linspace(-cz,np.shape(OGB)[2]-1-cz,np.shape(OGB)[2])
        zcoords = np.multiply(np.transpose(coords,(0,1,2)),zcoord)
        ogbz = np.multiply(OGB, np.transpose(-zcoords,(0,1,2)))
        ogbzz = np.multiply(OGB, np.transpose(zcoords**2,(0,1,2)))
        
        ogbxy = np.multiply(np.transpose(xcoords,(0,2,1)), -ogby)
        ogbxz = np.multiply(np.transpose(xcoords,(0,2,1)), -ogbz)
        ogbyz = np.multiply(np.transpose(ycoords,(2,1,0)), -ogbz)
        
        OGBX = [ogbx, ogby, ogbz]
        OGBXX = [[ogbxx, ogbxy, ogbxz],
                 [ogbxy, ogbyy, ogbyz],
                 [ogbxz, ogbyz, ogbzz]]
        
        if debug==1:
            print("Computed OGD, OGB, OGBX, OGBXX.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return OGD, OGB, OGBX, OGBXX
    
    elif variant==1:
        xcoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        ycoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        zcoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        
        cx=np.shape(OGB)[2] // 2
        cy=np.shape(OGB)[1] // 2
        cz=np.shape(OGB)[0] // 2 #this is all the same value in general
        
        #this is rather inefficient
        for z in range(np.shape(OGB)[0]):
            for y in range(np.shape(OGB)[1]):
                for x in range(np.shape(OGB)[2]):
                    xcoords[z,y,x]=x-cx
                    ycoords[z,y,x]=y-cy
                    zcoords[z,y,x]=z-cz 
        if debug==1:
            print("Computed OGD, OGB, xcoords,ycoords,zcoords.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return OGD, OGB, xcoords, ycoords, zcoords
    
def constructpcagrids_ocg(inpoccgrid, kr=6, variant=1, debug=0):
    """
    Construct all necessary occupancy grids for the calculation of the
    curvature estimation using integral invariants and PCA.
    
    Input:
        -'inp' is a strict or relaxed occupancy grid.
        -'kr' is the kernel radius
        -'variant': there are two variants to calculate these values, which
                    are equivalent, but return two different outputs.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Returns the Occupancy Grid of the Domain 'OGD', the Occupancy Grid of the
    Ball neighborhoord 'OGB', and the other two Occupancy Grids necessary for
    the convolution, which are OGB*(-x), here 'OGBX', and OGB*(x*x^T), here
    called 'OGBXX'.
    """
    if debug==1:
        starttime = time.time()    
        
    OGD = inpoccgrid
    
    OGB = constructoccgrid_ball(kr)
    
    if (np.shape(OGB)[0] %2 == 0 | 
        np.shape(OGB)[1] %2 == 0 | 
        np.shape(OGB)[2] %2 == 0):
        warnings.warn("The lengths of the kernel should be uneven s.t. there\
                      is a proper center element.")
                      
    if variant==0:
        ogbx, ogby, ogbz, ogbxx, ogbyy, ogbzz, ogbxy,\
        ogbxz, ogbyz = np.zeros([9, np.shape(OGB)[0], np.shape(OGB)[1], 
                                 np.shape(OGB)[2]])
        
        cx=np.shape(OGB)[1] // 2
        cy=np.shape(OGB)[0] // 2
        cz=np.shape(OGB)[2] // 2
        coords = np.ones([np.shape(OGB)[0],np.shape(OGB)[1],np.shape(OGB)[2]])
        
        xcoord = np.linspace(-cx,np.shape(OGB)[1]-1-cx,np.shape(OGB)[1])
        xcoords = np.multiply(np.transpose(coords,(0,2,1)),xcoord)
        ogbx = np.multiply(OGB, np.transpose(-xcoords,(0,2,1)))
        ogbxx = np.multiply(OGB, np.transpose(xcoords**2,(0,2,1)))
        
        ycoord = np.linspace(-cy,np.shape(OGB)[0]-1-cy,np.shape(OGB)[0])
        ycoords = np.multiply(np.transpose(coords,(2,1,0)),ycoord)
        ogby = np.multiply(OGB, np.transpose(-ycoords,(2,1,0)))
        ogbyy = np.multiply(OGB, np.transpose(ycoords**2,(2,1,0)))
        
        zcoord = np.linspace(-cz,np.shape(OGB)[2]-1-cz,np.shape(OGB)[2])
        zcoords = np.multiply(np.transpose(coords,(0,1,2)),zcoord)
        ogbz = np.multiply(OGB, np.transpose(-zcoords,(0,1,2)))
        ogbzz = np.multiply(OGB, np.transpose(zcoords**2,(0,1,2)))
        
        ogbxy = np.multiply(np.transpose(xcoords,(0,2,1)), -ogby)
        ogbxz = np.multiply(np.transpose(xcoords,(0,2,1)), -ogbz)
        ogbyz = np.multiply(np.transpose(ycoords,(2,1,0)), -ogbz)
        
        OGBX = [ogbx, ogby, ogbz]
        OGBXX = [[ogbxx, ogbxy, ogbxz],
                 [ogbxy, ogbyy, ogbyz],
                 [ogbxz, ogbyz, ogbzz]]
        
        if debug==1:
            print("Computed OGD, OGB, OGBX, OGBXX.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return OGD, OGB, OGBX, OGBXX
    
    elif variant==1:
        xcoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        ycoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        zcoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        
        cx=np.shape(OGB)[2] // 2
        cy=np.shape(OGB)[1] // 2
        cz=np.shape(OGB)[0] // 2 #this is all the same value in general
        
        #this is rather inefficient
        for z in range(np.shape(OGB)[0]):
            for y in range(np.shape(OGB)[0]):
                for x in range(np.shape(OGB)[0]):
                    xcoords[z,y,x]=x-cx
                    ycoords[z,y,x]=y-cy
                    zcoords[z,y,x]=z-cz 
        if debug==1:
            print("Computed OGD, OGB, xcoords,ycoords,zcoords.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return OGD, OGB, xcoords, ycoords, zcoords
      
def constructpcagrids_ms_pointcloud(inp, kr, rho, startscale, scaledist, 
                                    ocg="str", taulow=0, variant=1,
                                    debug=0):
    """
    Construct all necessary occupancy grids for the calculation of the
    multiscale curvature estimation using integral invariants and PCA.
    The main difference is the splitting of the kernel into separate parts.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         coordinates x y z or a string that leads to the file that is in x y z
         format with no header. 
        -'rho' controls the amount of cells in the occupancy grid.
        -'kr' is the kernel radius
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the 
         usage of a relaxed occupancy grid.
        -'variant': there are two variants to calculate these values, which
                    are equivalent, but return two different outputs.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Returns the Occupancy Grid of the Domain 'OGD', the Occupancy Grid of the
    Ball neighborhoord 'OGB', and the other two Occupancy Grids necessary for
    the convolution, which are OGB*(-x), here 'OGBX', and OGB*(x*x^T), here
    called 'OGBXX'.
    """
    if debug==1:
        starttime = time.time()    
        
    if type(inp)==str:
        #load pointcloud
        if ocg=="str":
            x, y, z = np.loadtxt(inp,skiprows=1, unpack=True,
                             usecols=(0,1,2))
        else:
            x, y, z, vals = np.loadtxt(inp,skiprows=1, unpack=True,
                             usecols=(0,1,2,3))
    elif isinstance(inp,np.ndarray):
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]
        
        if ocg=="rlx":
            vals = inp[:,3]
    elif isinstance(inp,list):
        #we assume the input list is given like [z,y,x]
        inp = np.transpose(inp) # shape is now [#pts, 3] (= z y x)

        #get the separate coordinate vectors
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]
        
        if ocg=="rlx":
            vals = inp[:,3]
    else:
        raise NameError('Input can be an already loaded pointcloud that \
                        consists of the three coordinates z y x or a string \
                        that leads to the file that is in z y x format with \
                        no header.')
                        
    if debug==1:
        print("Initialised the input point cloud.\n"
              + "Current Runtime: " + str(time.time() - starttime))
        print("Number of Points in the pointcloud: " + str(np.shape(x)[0]))
    
    if ocg=="str":
        OGD = constructoccgrid_pointcloud([z,y,x], rho)
    else:
        OGD = constructoccgrid_pointcloud([z,y,x, vals], rho, ocg=ocg, 
                                          taulow=taulow)
    
    if kr==startscale:
        OGB = constructoccgrid_ball(kr)
    else:
        OGB = constructoccgrid_ballzone(kr=kr, alpha=scaledist)
    
    if (np.shape(OGB)[0] %2 == 0 | 
        np.shape(OGB)[1] %2 == 0 | 
        np.shape(OGB)[2] %2 == 0):
        warnings.warn("The lengths of the kernel should be uneven s.t. there\
                      is a proper center element.")
                      
    if variant==0:
        ogbx, ogby, ogbz, ogbxx, ogbyy, ogbzz, ogbxy,\
        ogbxz, ogbyz = np.zeros([9, np.shape(OGB)[0], np.shape(OGB)[1], 
                                 np.shape(OGB)[2]])
        
        cx=np.shape(OGB)[1] // 2
        cy=np.shape(OGB)[0] // 2
        cz=np.shape(OGB)[2] // 2
        coords = np.ones([np.shape(OGB)[0],np.shape(OGB)[1],np.shape(OGB)[2]])
        
        xcoord = np.linspace(-cx,np.shape(OGB)[1]-1-cx,np.shape(OGB)[1])
        xcoords = np.multiply(np.transpose(coords,(0,2,1)),xcoord)
        ogbx = np.multiply(OGB, np.transpose(-xcoords,(0,2,1)))
        ogbxx = np.multiply(OGB, np.transpose(xcoords**2,(0,2,1)))
        
        ycoord = np.linspace(-cy,np.shape(OGB)[0]-1-cy,np.shape(OGB)[0])
        ycoords = np.multiply(np.transpose(coords,(2,1,0)),ycoord)
        ogby = np.multiply(OGB, np.transpose(-ycoords,(2,1,0)))
        ogbyy = np.multiply(OGB, np.transpose(ycoords**2,(2,1,0)))
        
        zcoord = np.linspace(-cz,np.shape(OGB)[2]-1-cz,np.shape(OGB)[2])
        zcoords = np.multiply(np.transpose(coords,(0,1,2)),zcoord)
        ogbz = np.multiply(OGB, np.transpose(-zcoords,(0,1,2)))
        ogbzz = np.multiply(OGB, np.transpose(zcoords**2,(0,1,2)))
        
        ogbxy = np.multiply(np.transpose(xcoords,(0,2,1)), -ogby)
        ogbxz = np.multiply(np.transpose(xcoords,(0,2,1)), -ogbz)
        ogbyz = np.multiply(np.transpose(ycoords,(2,1,0)), -ogbz)
        
        OGBX = [ogbx, ogby, ogbz]
        OGBXX = [[ogbxx, ogbxy, ogbxz],
                 [ogbxy, ogbyy, ogbyz],
                 [ogbxz, ogbyz, ogbzz]]
        
        if debug==1:
            print("Computed OGD, OGB, OGBX, OGBXX.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return OGD, OGB, OGBX, OGBXX
    
    elif variant==1:
        xcoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        ycoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        zcoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        
        cx=np.shape(OGB)[2] // 2
        cy=np.shape(OGB)[1] // 2
        cz=np.shape(OGB)[0] // 2 #this is all the same value in general
        
        #this is rather inefficient
        for z in range(np.shape(OGB)[0]):
            for y in range(np.shape(OGB)[0]):
                for x in range(np.shape(OGB)[0]):
                    xcoords[z,y,x]=x-cx
                    ycoords[z,y,x]=y-cy
                    zcoords[z,y,x]=z-cz 
        if debug==1:
            print("Computed OGD, OGB, xcoords,ycoords,zcoords.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return OGD, OGB, xcoords, ycoords, zcoords

def constructpcagrids_ms_ocg(inpocg, kr, startscale, scaledist,
                             variant=1, debug=0):
    """
    Construct all necessary occupancy grids for the calculation of the
    multiscale curvature estimation using integral invariants and PCA.
    The main difference is the splitting of the kernel into separate parts.
    
    Input:
        -'inp' is a strict or relaxed occupancy grid
        -'kr' is the kernel radius
        -'variant': there are two variants to calculate these values, which
                    are equivalent, but return two different outputs.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Returns the Occupancy Grid of the Domain 'OGD', the Occupancy Grid of the
    Ball neighborhoord 'OGB', and the other two Occupancy Grids necessary for
    the convolution, which are OGB*(-x), here 'OGBX', and OGB*(x*x^T), here
    called 'OGBXX'.
    """
    if debug==1:
        starttime = time.time()    
        
    OGD = inpocg
    
    if kr==startscale:
        OGB = constructoccgrid_ball(kr)
    else:
        OGB = constructoccgrid_ballzone(kr=kr, alpha=scaledist)
    
    if (np.shape(OGB)[0] %2 == 0 | 
        np.shape(OGB)[1] %2 == 0 | 
        np.shape(OGB)[2] %2 == 0):
        warnings.warn("The lengths of the kernel should be uneven s.t. there\
                      is a proper center element.")
                      
    if variant==0:
        ogbx, ogby, ogbz, ogbxx, ogbyy, ogbzz, ogbxy,\
        ogbxz, ogbyz = np.zeros([9, np.shape(OGB)[0], np.shape(OGB)[1], 
                                 np.shape(OGB)[2]])
        
        cx=np.shape(OGB)[1] // 2
        cy=np.shape(OGB)[0] // 2
        cz=np.shape(OGB)[2] // 2
        coords = np.ones([np.shape(OGB)[0],np.shape(OGB)[1],np.shape(OGB)[2]])
        
        xcoord = np.linspace(-cx,np.shape(OGB)[1]-1-cx,np.shape(OGB)[1])
        xcoords = np.multiply(np.transpose(coords,(0,2,1)),xcoord)
        ogbx = np.multiply(OGB, np.transpose(-xcoords,(0,2,1)))
        ogbxx = np.multiply(OGB, np.transpose(xcoords**2,(0,2,1)))
        
        ycoord = np.linspace(-cy,np.shape(OGB)[0]-1-cy,np.shape(OGB)[0])
        ycoords = np.multiply(np.transpose(coords,(2,1,0)),ycoord)
        ogby = np.multiply(OGB, np.transpose(-ycoords,(2,1,0)))
        ogbyy = np.multiply(OGB, np.transpose(ycoords**2,(2,1,0)))
        
        zcoord = np.linspace(-cz,np.shape(OGB)[2]-1-cz,np.shape(OGB)[2])
        zcoords = np.multiply(np.transpose(coords,(0,1,2)),zcoord)
        ogbz = np.multiply(OGB, np.transpose(-zcoords,(0,1,2)))
        ogbzz = np.multiply(OGB, np.transpose(zcoords**2,(0,1,2)))
        
        ogbxy = np.multiply(np.transpose(xcoords,(0,2,1)), -ogby)
        ogbxz = np.multiply(np.transpose(xcoords,(0,2,1)), -ogbz)
        ogbyz = np.multiply(np.transpose(ycoords,(2,1,0)), -ogbz)
        
        OGBX = [ogbx, ogby, ogbz]
        OGBXX = [[ogbxx, ogbxy, ogbxz],
                 [ogbxy, ogbyy, ogbyz],
                 [ogbxz, ogbyz, ogbzz]]
        
        if debug==1:
            print("Computed OGD, OGB, OGBX, OGBXX.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return OGD, OGB, OGBX, OGBXX
    
    elif variant==1:
        xcoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        ycoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        zcoords = np.zeros([OGB.shape[0],OGB.shape[1],OGB.shape[2]])
        
        cx=np.shape(OGB)[2] // 2
        cy=np.shape(OGB)[1] // 2
        cz=np.shape(OGB)[0] // 2 #this is all the same value in general
        
        #this is rather inefficient
        for z in range(np.shape(OGB)[0]):
            for y in range(np.shape(OGB)[0]):
                for x in range(np.shape(OGB)[0]):
                    xcoords[z,y,x]=x-cx
                    ycoords[z,y,x]=y-cy
                    zcoords[z,y,x]=z-cz 
        if debug==1:
            print("Computed OGD, OGB, xcoords,ycoords,zcoords.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return OGD, OGB, xcoords, ycoords, zcoords
    
def constructboundary(inp, sigma=0.4):
    '''
    Calculates boundary mask of matrix inp and returns it. This is sometimes
    a bit unstable and requires the manual use of the binary dilation method.
    
    If you want to get a "strict" boundary, i.e. really just the exact 
    boundary, then set sigma to 0. If you want to get a slightly smoothed
    boundary, setting sigma to 0.4 usually works well.
    
    The reason why this method exists in the first place is to make the
    visualisations easier. Sometimes the boundaries have holes or can be
    seen through at points, which is why the sigma parameter exists.
    '''
    mask = inp - nd.morphology.binary_dilation(inp)
    helper = mask - nd.morphology.binary_dilation(inp)
    
    out = (nd.gaussian_filter(255*((helper*inp)<0), 
                              sigma=sigma)/255 !=0).astype(int)
    
    return out

def constructoccgrid_imgstack(imgstack, stacktype, rho=0, thr=0, ocg='str', 
                              fill=1, optional=[0,0]):
    """
    constructs a strict or relaxed occupancy grid from a dataset of stacked
    images such as .tiff or .h5 .

    Parameters
    ----------
    imgstack : str or np.array
        path to file/folder as string or numpy array of dataset.
    stacktype : str
        There are four available types:
            -'path2tiffolder': means imgstack contains path to the folder full
                               of slices.
            -'path2tiff': means imgstack contains path to a .tiff file
            -'path2h5': means imgstack contains path to an .h5 file
            -'nparray': means imgstack is a numpy array that contains all
                        image stacks
    rho : float, optional
        size parameter that specifies how large the respective image slices
        are. The default is 0, which means the size of the original dataset is
        used. For large datasets, setting rho!=0 leads to a massive increase
        in runtime, it's better to decrease the size of the slices by
        different means.
    thr : float, optional
        threshold parameter. Values equal or below are disregarded. For strict
        occupancy grids, a values above are set to one in the occupancy grid.
        For relaxed occupancy grids, all values above are directly put into the 
        occupancy grid. The default is 0.
    ocg : str, optional
        two choices: 'str' or 'rlx'. Specifies the kind of occupancy grid that
        the method outputs. 'str' means a strict occupancy grid is output, 
        'rlx' means a relaxed occupancy grid is output. The default is 'str'.
    fill : bool, optional
        specifies whether the strict occupancy grid shall be filled or not. If
        a strict occupancy grid is not filled, the curvature estimation
        framework is not well-defined. The default is 1.
    optional : list, optional
        serves as an optional parameter for the path2folder and path2h5 
        stacktypes. Can be used to cut down the amount of slices and works
        differently for each stacktype. For path2folder, the first element
        specifies how many slices are removed from the start and the second
        element how many are removed rom the back. For path2h5, the two
        numbers specify an interval of slices that are used, for example 
        slices 20 to 30, then optional=[20,30]. The default is [0,0], which
        means the original datasets is used.

    Raises
    ------
    NameError
        if a wrong stacktype is used.

    Returns
    -------
    ocg : numpy array
        strict or relaxed occupancy grid as a numpy array.

    """
    #read different input image stacks and transform them all into a numpy
    #array (except for hdf5 format, then we read it as hdf5 and keep the
    #references)
    #path2folder
    if stacktype == 'path2tiffolder':
        #get all files in folder
        filelist = []
            #assume all files in folder are slices
        for root, dirs, files in os.walk(imgstack):
            for file in files:
                filelist.append(os.path.join(root,file))
                
        #narrow down filelist, useful if folder contains huge amount of slices
        if optional[0]!=0 and optional[1]!=0:
            for i in range(0,optional[0]):
                filelist.pop(0)
            for i in range(0,optional[1]):
                filelist.pop()
                
        #go through all files and put them into the nparray
        z = len(filelist)
        img = Image.open(filelist[0])#assumes all slices have same size
        dims = np.shape(img)
        imgstack = np.zeros([z,dims[0],dims[1]])
        counter = 0
        for file in filelist:
            img = Image.open(file)
            imgstack[counter,:,:] = img
            counter = counter+1
    #path2tiff
    elif stacktype == 'path2tiff':
        imgstack = Image.open(imgstack)#read tiff and tif files
        imgstack = np.array(imgstack)
    #path2h5
    elif stacktype == 'path2h5':
        imgstack = h5py.File(imgstack, 'r')
        keys = list(imgstack.keys())
        #we assume there is only one key, if this is not the case, adapt
        #the following line accordingly
        imgstack = imgstack.get(keys[0])
        if optional[0]!=0 and optional[1]!=0:
            imgstack = imgstack[optional[0]:optional[1]+1,:,:]
    #array
    elif stacktype == 'nparray':
        #do nothing
        pass
    else:
        raise NameError('Unknown type of image stack.')
    
    dims = np.shape(imgstack)
    
    if rho!=0:
        ocg = np.zeros([int(dims[0]*rho)+1,int(dims[1]*rho)+1,
                        int(dims[2]*rho)+1])
        for i in range(0,dims[0]):#this takes veeery long on big datasets
            for ii in range(0,dims[1]):
                for iii in range(0,dims[2]):
                    if imgstack[i,ii,iii]>thr:
                        if ocg=='str':
                            ocg[int(i * rho), int(ii* rho),int(iii * rho)] = 1
                            if fill==1:
                                ocg = nd.binary_fill_holes(ocg).astype(int)
                        else:
                            ocg[int(i * rho), int(ii* rho),
                                int(iii * rho)] = imgstack[i,ii,iii]
    else:
        if ocg=='str':
            ocg = np.zeros([dims[0],dims[1],dims[2]])
            finder = np.where(imgstack>thr)
            ocg[finder] = 1
            if fill==1:
                ocg = nd.binary_fill_holes(ocg).astype(int)
        else:
            if thr==0:
                ocg = imgstack
            else:
                ocg = np.zeros([dims[0],dims[1],dims[2]])
                finder = np.where(imgstack>thr)
                ocg[finder] = imgstack[finder]
    
    return ocg
  
    
  
#end