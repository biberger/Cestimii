# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:13:44 2020

@author: Simon Biberger
"""

import numpy as np

def circle(r=1, c=[0,0,0], sampH=360):
    """
    Returns parametrization of a circle with radius 'r' at 
    center (cx,cy,cz) and 'sampH' amount of samples.
    
    The default values are 1, (0,0,0) and 360 for the radius, base point,
    and amount of samples, respectively.
    """
    cx, cy, cz = c
    theta = np.linspace(0, 2*np.pi, sampH)
    x = r* np.cos(theta) - cx
    y = r * np.sin(theta) - cy
    z = cz * np.ones(theta.shape)
    return x, y, z

def filledcircle(r=1, c=[0,0,0], sampH=360, fcirc=20):
    """
    Returns parametrization of a filled circle with radius 'r' at 
    center (cx,cy,cz), with 'sampH' amount of horizontal samples,
    i.e. samples of the circle, and 'fcirc' amount of circles that fill the outer circle of radius 'r'.
    
    The default values are 1, (0,0,0), 360, and 20 for the radius,
    center, horizontal samples, and filling circles, respectively.
    """
    cx, cy, cz = c
    theta = np.linspace(0, 2*np.pi, sampH)
    r = np.linspace(0,r,fcirc)
    theta, r = np.meshgrid(theta,r)

    x= r * np.cos(theta) - cx
    y = r * np.sin(theta) - cy
    z = cz * np.ones(theta.shape)
    return x, y, z

def opencylinder(r=1, h=2, bp=[0,0,0], sampH=360, sampV=50):
    """
    Returns parametrization of an open (i.e. pipelike) cylinder with
    radius 'r', height 'h', base point (bpx,bpy,bpz), 
    where 'sampH' and 'sampV' specify the amount of samples used 
    horizontally, i.e. for circles, and vertically, i.e. for height.
    The base point is in the cylinder's center at the bottom.
    
    The default values for the radius, height, center, 'sampH' and 'sampV' 
    are 1, 2, (0,0,0), 360 and 50, respectively.
    """
    bpx, bpy, bpz = bp
    theta = np.linspace(0, 2*np.pi, sampH)
    z = np.linspace(bpz, bpz+h, sampV)
    theta, z = np.meshgrid(theta, z)
    x = r * np.cos(theta) - bpx
    y = r * np.sin(theta)- bpy
    return x, y, z

def opencylinder2(r=1, h=2, bp=[0,0,0], sampH=0.1, sampV=0.1):
    """
    Returns parametrization of an open (i.e. pipelike) cylinder with
    radius 'r', height 'h', base point (bpx,bpy,bpz), 
    where 'sampH' and 'sampV' specify the amount of samples used 
    horizontally, i.e. for circles, and vertically, i.e. for height.
    The base point is in the cylinder's center at the bottom.
    
    The default values for the radius, height, center, 'sampH' and 'sampV' 
    are 1, 2, (0,0,0), 360 and 50, respectively.
    
    Difference to opencylinder() is the use of np.arange instead 
    of np.linspace in the initialization of theta.
    """
    bpx, bpy, bpz = bp
    theta = np.arange(0, 2*np.pi, sampH)
    z = np.arange(bpz, bpz+h, sampV)
    theta, z = np.meshgrid(theta, z)
    x = r * np.cos(theta) - bpx
    y = r * np.sin(theta)- bpy
    return x, y, z

def closedcylinder(r=1, h=2, bp=[0,0,0], sampH=360, sampV=50, fcirc=20):
    """
    Returns the parametrization of a closed cylinder with radius 'r',
    height 'h', base point (bpx,bpy,bpz), where 'sampH', 'sampV' specify 
    the amount of samples used horizontally, i.e. for circles,
    vertically, i.e. for height, and 'fcirc' specifies the amount 
    of circles that fill the outer perimeter of radius 'r' of the cylinder,
    i.e. how many circles are used to fill the bottom and top.
    The base point is in the cylinder's center at the bottom.
    
    The default values for the radius, height, center, 'sampH', 'sampV',
    'fcirc' are 1, 2, (0,0,0), 360, 50, and 20, respectively.
    """
    bpx, bpy, bpz = bp
    theta = np.linspace(0, 2*np.pi, sampH)
    z = np.linspace(bpz, bpz+h, sampV )
    theta, z = np.meshgrid(theta, z)
    x = r * np.cos(theta) - bpx
    y = r * np.sin(theta) - bpy
    
    xcirc, ycirc, zcirc = filledcircle(r=r,c=[bpx,bpy,bpz], sampH=sampH,
                                       fcirc=fcirc)
    
    x = np.append(x,xcirc,0)
    y = np.append(y,ycirc,0)
    z = np.append(z,zcirc,0)
    
    x = np.append(x,xcirc,0)
    y = np.append(y,ycirc,0)
    z = np.append(z,zcirc+h,0)

    return x, y, z

def sphere(r=1, c=[0,0,0], samp=360):
    """
    Returns parametrization of a sphere with radius 'r' at 
    center (cx,cy,cz) and 'samp' amount of samples in all directions.
    
    The default values are 1, (0,0,0) and 360 for the radius, center,
    and amount of samples, respectively.
    """
    cx, cy, cz = c
    theta = np.linspace(0, 2*np.pi, samp)
    phi = np.linspace(0, np.pi, samp)
    
    phi, theta = np.meshgrid(phi, theta)
    #z = cz*np.ones(theta.shape)
    
    r_xy = r*np.sin(phi)
    x = cx + r_xy * np.cos(theta) 
    y = cy + r_xy * np.sin(theta)
    z = cz + r * np.cos(phi)
    
    return x, y, z

def opencone(r=1, h=5, bp=[0,0,0], sampH=360, sampV=50):
    """
    Returns parametrization of an open cone with radius 'r' and height 'h at 
    basepoint (bpx,bpy,bpz), where 'sampH' and 'sampV' specify the amount of 
    samples used horizontally, i.e. for circles, and vertically, i.e. 
    for height. The base point is in the cones's center at the bottom.
    
    The default values are 1, 5, (0,0,0), 360 and 50 for the radius, center,
    and amount of horizontal and vertical samples, respectively.
    """
    bpx, bpy, bpz = bp
    theta0 = np.linspace(0, 2*np.pi, sampH)
    z = np.linspace(bpz, bpz+h, sampV)
    theta, z = np.meshgrid(theta0, z)
    r = np.linspace(r, 0, sampV)
    theta, r = np.meshgrid(theta0, r)
    
    x = r * np.cos(theta) - bpx
    y = r * np.sin(theta) - bpy
    
    return x, y, z

def closedcone(r=1, h=5, bp=[0,0,0], sampH=360, sampV=50, fcirc=20):
    """
    Returns parametrization of a closed cone with radius 'r' and height 'h at 
    basepoint (bpx,bpy,bpz), where 'sampH' and 'sampV' specify the amount of 
    samples used horizontally, i.e. for circles, and vertically, i.e. 
    for height, and 'fcirc' specifies the amount 
    of circles that fill the bottom of the cone with radius 'r',
    The base point is in the cones's center at the bottom.
    
    The default values are 1, 5, (0,0,0), 360 and 50 for the radius, center,
    and amount of horizontal and vertical samples, respectively.
    """
    bpx, bpy, bpz = bp
    theta0 = np.linspace(0, 2*np.pi, sampH)
    z = np.linspace(bpz, bpz+h, sampV)
    theta, z = np.meshgrid(theta0, z)
    r = np.linspace(r, 0, sampV)
    theta, r = np.meshgrid(theta0, r)
    
    x = r * np.cos(theta) - bpx
    y = r * np.sin(theta) - bpy
    
    xcirc, ycirc, zcirc = filledcircle(r=r,c=[bpx,bpy,bpz], sampH=sampH,
                                       fcirc=fcirc)
    
    x = np.append(x,xcirc,0)
    y = np.append(y,ycirc,0)
    z = np.append(z,zcirc,0)
    
    return x, y, z

def testshape_ocg():
    '''
    Returns the occupancy grid of a testshape used in some parts of the 
    thesis.
    '''
    W=201
    H=20
    total=np.zeros((W,W,W), np.int8)
    for i in range(H):
        total[i,0:W,0:W]=1
    for i in range(W//2):
        total[W//2+H-1-i,0:W,W//2-i:W//2+i+1]=1
    radius = 75
    R = 2*radius +1
    center=(radius, radius, radius)
    a=[]
    b=[]
    c=[]
    diff=W-R
    for z in range(R):
        for y in range(R):
            for x in range(R):
                val=(x-center[2])**2+(y-center[1])**2+(z-center[0])**2
                if val<=radius**2:
                    a.append(x)
                    b.append(y)
                    c.append(z)
                    total[diff//2+z+H-1,diff//2+y,diff//2+x]=1
    return total