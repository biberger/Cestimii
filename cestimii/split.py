# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:28:58 2021

@author: Simon Biberger

Cestimii.
Compute curvature estimations on point clouds using integral invariants.

This file contains all methods that have something to do with the calculation 
of curvature estimations, such as principal curvatures, mean curvature,
gaussian curvature, etc. .
"""

import time
import numpy as np

from math import ceil, log

def spbs_handleinput_pointcloud(inp, lex, rg, ocg, debug=0):
    """
    Helper function for spbs. Handles the input (pointcloud) and returns x,y,z
    coordinates (in this order) as vectors (nx1) if ocg="str". If ocg='rlx',
    we also return the probability/intensity values 'prob' in a n by 1 vector.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         or four coordinates x y z or x y z intensity or a string that leads 
         to the file that is in x y z or x y z intensity format with no
         header. 
        -'lex' specifies wether the point cloud is already in lexicographic
         order(==1) or not (==0) and if there are duplicate points in the
         point cloud or not (==0, means there might be duplicate values,
         otherwise it is assumed that there aren't).
        -'kr' is the kernel radius.
        -'rg' is the parameter that specifies wether the input grid is
         regular or not and has reasonable values (e.g. not 1.1, but 110).
        -'ocg' is used to control which occgrid will be used later on. Set
         ocg='str' for the strict occgrid and ocg='rlx' for the relaxed one.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
         
    Details on 'rg'. If we don't want any rescaling/"downsampling", set rg!=1.
    Otherwise, the parameter is similar to the useage of rho in the occgrid
    methods. For this, set rg=[~, rho] where ~ can be anything, but rho is
    the factor used in the rescaling.
    
    Returns the x,y,z coordinates (in this order) as vectors (n by 1, resp.).
    If ocg='rlx', then we also return the probability/intensity values 'prob' 
    in a n by 1 vector.
    """
    if debug==1:
        starttime = time.time()
        
    if rg!=1:
        if len(rg)!=2:
            raise NameError('Parameter rg specifies wether the input is on a \
                            regular grid or not, and if it has reasonable \
                            values or not. If rg==1, the input is supposed \
                            to have a regular grid and reasonable values. \
                            In the other case, we have to deal with special \
                            circumstances, which require a rescaling \
                            parameter rho (in the same way as it was used \
                            in the old occgrid method). I.e. the input shall \
                            be a [0,rho]. Example: If xmax is 0.9, choosing \
                            rho=10 or 100 should give good results.')
        else:
            rgp, rho = rg
    else:
        rho = 1 #i.e. there won't be any rescaling

    if ocg == "str":
        if type(inp)==str:
            #load pointcloud
            x, y, z = np.loadtxt(inp,skiprows=1, unpack=True,
                                 usecols=(0,1,2))
            
            #find min,max values of each axis
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            zmin = np.min(z)
            zmax = np.max(z)
            
            #normalise x,rescale axes to max val (both if irreg grid) and conform
            #to regular grid values by making it int. This also has the nice
            #effect of reducing the volume to its actual values(i.e. no borders of 
            #zeros or sth)
            xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
            yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
            zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)
            
            #force lexicographic order
            '''
            I.e. list with elements that consist of [x,y,z] coordinates is sorted
            from z_min to z_max. And for a fix z, it is sorted y_min to y_max.
            And for a fix y, it is sorted x_min to x_max.
            '''
            if lex==0:
                inp = np.transpose([zrs,yrs,xrs]) # shape is [#pts, 3] (= z y x)
                inp = inp.tolist()
                inp = sorted(inp) #get lex order
                
                #remove duplicate points (as inp is not hashable this is
                #a bit slow :// )
                for i in range(len(inp)-1,0,-1): #uses the lexorder
                    if inp[i]==inp[i-1]:
                        inp.pop(i)
                    
                inp = np.asarray(inp)
                
                #get the separate coordinate vectors
                zrs = inp[:,0]
                yrs = inp[:,1]
                xrs = inp[:,2]
            else:
                inp = np.transpose([zrs,yrs,xrs])
                
                #get the separate coordinate vectors
                zrs = inp[:,0]
                yrs = inp[:,1]
                xrs = inp[:,2]
            
        elif isinstance(inp,np.ndarray):
            z = inp[:,0]
            y = inp[:,1]
            x = inp[:,2]
            
            #find min,max values of each axis
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            zmin = np.min(z)
            zmax = np.max(z)
            
            #normalise x,y,z and then rescale axes to max val
            xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
            yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
            zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)
            
            inp = np.transpose([zrs,yrs,xrs])
            
            #force lexicographic order
            if lex==0:
                #we assume the input nd array is given in
                #z y x format
                inp = inp.tolist()
                inp = sorted(inp)
                
                #remove duplicate points (as inp is not hashable this is
                #a bit slow :// )
                for i in range(len(inp)-1,0,-1): #uses the lexorder
                    if inp[i]==inp[i-1]:
                        inp.pop(i)
                
                #initialize event point queue
                inp = np.asarray(inp)
                
                #get the separate coordinate vectors
                zrs = inp[:,0]
                yrs = inp[:,1]
                xrs = inp[:,2]
            
        elif isinstance(inp,list):
            #we assume the input list is given like [z,y,x]
            inp = np.transpose(inp) # shape is now [#pts, 3] (= z y x)
            
            #force lexicographic order
            if lex==0:
                inp = inp.tolist()
                inp = sorted(inp)
                
                #remove duplicate points (as inp is not hashable this is
                #a bit slow :// )
                for i in range(len(inp)-1,0,-1): #uses the lexorder
                    if inp[i]==inp[i-1]:
                        inp.pop(i)
                
                inp = np.asarray(inp)
            
                
            #get the separate coordinate vectors
            z = inp[:,0]
            y = inp[:,1]
            x = inp[:,2]
            
            #find min,max values of each axis
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            zmin = np.min(z)
            zmax = np.max(z)
            
            #normalise x and then rescale axes to max val
            xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
            yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
            zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)
            
            #inp = np.transpose([zrs,yrs,xrs])
            
        else:
            raise NameError('Input can be an already loaded pointcloud that \
                            consists of the three coordinates z y x or a string \
                            that leads to the file that is in z y x format with \
                            no header.')
        return xrs,yrs,zrs
    
    elif ocg=="rlx":
        if type(inp)==str:
            #load pointcloud
            x, y, z, prob = np.loadtxt(inp,skiprows=1, unpack=True,
                                 usecols=(0,1,2,3))
            
            #find min,max values of each axis
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            zmin = np.min(z)
            zmax = np.max(z)
            
            #normalise x,rescale axes to max val (both if irreg grid) and conform
            #to regular grid values by making it int. This also has the nice
            #effect of reducing the volume to its actual values(i.e. no borders of 
            #zeros or sth)
            xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
            yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
            zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)
            
            #force lexicographic order
            '''
            I.e. list with elements that consist of [x,y,z] coordinates is sorted
            from z_min to z_max. And for a fix z, it is sorted y_min to y_max.
            And for a fix y, it is sorted x_min to x_max.
            '''
            if lex==0:
                inp = np.transpose([zrs,yrs,xrs,prob]) # shape is [#pts, 3] (= z y x)
                inp = inp.tolist()
                inp = sorted(inp) #get lex order
                
                #remove duplicate points (as inp is not hashable this is
                #a bit slow :// )
                for i in range(len(inp)-1,0,-1): #uses the lexorder
                    if inp[i]==inp[i-1]:
                        inp.pop(i)
                    
                inp = np.asarray(inp) #this is a terrible way to do it, but hey
                
                #get the separate coordinate vectors
                zrs = inp[:,0]
                yrs = inp[:,1]
                xrs = inp[:,2]
                prob = inp[:,3]
            else:
                inp = np.transpose([zrs,yrs,xrs,prob])
                
                #get the separate coordinate vectors
                zrs = inp[:,0]
                yrs = inp[:,1]
                xrs = inp[:,2]
                prob = inp[:,3]
            
        elif isinstance(inp,np.ndarray):
            z = inp[:,0]
            y = inp[:,1]
            x = inp[:,2]
            prob = inp[:,3]
            
            #find min,max values of each axis
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            zmin = np.min(z)
            zmax = np.max(z)
            
            #normalise x,y,z and then rescale axes to max val
            xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
            yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
            zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)
            
            inp = np.transpose([zrs,yrs,xrs,prob])
            
            #force lexicographic order
            if lex==0:
                #we assume the input nd array is given in
                #z y x prob format
                inp = inp.tolist()
                inp = sorted(inp)
                
                #remove duplicate points (as inp is not hashable this is
                #a bit slow :// )
                for i in range(len(inp)-1,0,-1): #uses the lexorder
                    if inp[i]==inp[i-1]:
                        inp.pop(i)
                
                #initialize event point queue
                inp = np.asarray(inp)
                
                #get the separate coordinate vectors
                zrs = inp[:,0]
                yrs = inp[:,1]
                xrs = inp[:,2]
                prob = inp[:,3]
            
        elif isinstance(inp,list):
            #we assume the input list is given like [z,y,x]
            inp = np.transpose(inp) # shape is now [#pts, 3] (= z y x)
            
            #force lexicographic order
            if lex==0:
                inp = inp.tolist()
                inp = sorted(inp)
                
                #remove duplicate points (as inp is not hashable this is
                #a bit slow :// )
                for i in range(len(inp)-1,0,-1): #uses the lexorder
                    if inp[i]==inp[i-1]:
                        inp.pop(i)
                
                inp = np.asarray(inp)
            
                
            #get the separate coordinate vectors
            z = inp[:,0]
            y = inp[:,1]
            x = inp[:,2]
            prob = inp[:,3]
            
            #find min,max values of each axis
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            zmin = np.min(z)
            zmax = np.max(z)
            
            #normalise x and then rescale axes to max val
            xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
            yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
            zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)
            
            #inp = np.transpose([zrs,yrs,xrs])
            
        else:
            raise NameError('Input can be an already loaded pointcloud that \
                            consists of the three coordinates z y x or a string \
                            that leads to the file that is in z y x format with \
                            no header.')
        return xrs,yrs,zrs,prob
    else:
        raise NameError('Wrong ocg parameter. Only str or rlx allowed.')
    
    if debug==1:
        print("Initialised the input point cloud in lex. order.\n"
              + "Current Runtime: " + str(time.time() - starttime))
        print("Number of Points in the pointcloud: " + str(np.shape(xrs)[0]))

def spbs_handleinput_ocg(inp, ocg=0):
    """
    Helper function for spbs. Handles the input (occgrid) and returns x,y,z
    coordinates (in this order) as vectors (nx1) if ocg="str". If ocg='rlx',
    we also return the probability/intensity values 'prob' in a n by 1 vector.
    
    Input:
        -'inp' can be strict or relaxed occupancy grid
        -'ocg' decides wether input occupancy grid is handled as a strict or
         relaxed occupancy grid. ==0 for strict, ==1 for relaxed.
    
    Returns the x,y,z coordinates (in this order) as vectors (n by 1, resp.).
    If inp is a relaxed occupancy grid, then we also return the  
    probability/intensity values 'prob' in a n by 1 vector.
    """
    #get all relevant entries
    entries = np.where(inp>0)
    entries = np.transpose(np.asarray(entries))
    
    #force lex order (I'm not sure if we get the lex order automatically
    #through np.where. If that's the case, we could remove the part below
    #and get better performance)
    entries = entries.tolist()
    entries = sorted(entries) #get lex order
    entries= np.asarray(entries)
    
    #initialise coordinate vectors
    x = entries[:,2]
    y = entries[:,1]
    z = entries[:,0]
    
    if ocg==0:
        return x, y, z
    else:
        prob = np.zeros([np.shape(entries)[0],1])
        
        for i in range(0,np.shape(entries)[0]):
            prob[i] = inp[z[i], y[i], x[i]]
            
        return x, y, z, prob

def spbs_checkboxcollisions(newboxcntr, rsps, bbsize):
    '''
    Helper function for spbs. Checks the newbox for collisions with the
    neighboring boxes in rsps. Returns the [z,y,x] coordinates of the center
    of the new box. newboxcntr is given in z,y,x.
    '''
    z,y,x = newboxcntr
    
    #initialize/reset the collision flags and resp. coordinates
    collfrt = 0 #collfrt
    colltop = 0 #colltop(=collision of top of newbox with some neighboring box)
    collbot = 0 #bottom
    collrgh = 0 #right
    colllft = 0 #left
    
    collfree = 0
    
    if rsps: #i.e. if regional sweep splane status not empty
        #check for collisions until either we accept overlapping boxes
        #or all checks go through successfully
        while(collfree==0):
            #loop through all active boundary boxes and check each bb
            #for collisions with the "new" bb
            
            collfree = 1 #only if collisions inside the for-loop occur,
            #then we have to go through this while loop. Otherwise we stop.
            
            for actbb in rsps: #actbb for "active boundary box"
            #actbb indices: z,y,x
                
                #z-axis collision? yes, if zmaxnew>zminactbb
                if (z - actbb[0] > -bbsize):
                    
                    #y-axis collision? yes, if ymaxnew>yminactbb or
                    #yminnew<ymaxactbb
                    if((np.abs(y - actbb[1]) <= bbsize)):
                       
                        #x-axis collision? yes, if xmaxnew>xminactbb or
                        #xminnew<xmaxactbb
                        if((np.abs(x - actbb[2]) <= bbsize)):
                           
                            #if collision occurs, try pushing back the new bb
                            #first->else case. If that was tried, collfrt=1
                            #and we go into the if case below
                            if collfrt==1: 
                                if ((colltop==1) | (collbot==1)): #i.e. has
                                #the new bb been adjusted to the top or
                                #bottom yet? ->no? go to else and adjust
                                    if ((collrgh==1) | (colllft==1)): #i.e. has
                                    #the new bb been adjusted to the right
                                    #or left yet?->no? go to else and do so
                                        break
                                        #accept overlap :/ we've tried
                                        #everything we could! 
                                        #One thing
                                        #we could do now, is adjust the bb
                                        #to the right for consistency's
                                        #sake, but I don't think that's 
                                        #necessary.
                                    else:
                                        #xmaxnew>actbb[5] (xminactbb)
                                        if x - actbb[2] <= -(bbsize//2):#collright
                                            x = actbb[2] - bbsize
                                            collrgh = 1
                                            collfree = 0
                                            break#check all bbs for colls 
                                            #again now that I have adjusted 
                                            #the new bb
                                            
                                        #xminnew<actbb[4] (xmaxactbb)
                                        elif x - actbb[2] > bbsize//2:#colllft
                                            x = actbb[2] + bbsize
                                            colllft = 1
                                            collfree = 0
                                            break
                                else: #no, it has not been adjusted yet
                                #-> adjust it now.
                                    
                                    #ymaxnew>actbb[3] (yminactbb)
                                    if y - actbb[1] <= -(bbsize//2):#colltop
                                        y = actbb[1] - bbsize
                                        colltop = 1
                                        collfree = 0
                                        break#check all bbs for collisions 
                                        #again now that I have adjusted 
                                        #the new bb
                                        
                                    #yminnew<actbb[2] (ymaxactbb)
                                    elif y - actbb[1] > bbsize//2:#collbott
                                        y = actbb[1] + bbsize
                                        collbot = 1
                                        collfree = 0
                                        break
                            else: #set zmaxnew to zminactbb and adjust
                            #the rest
                                z = actbb[0] - bbsize #collfrt, move bb back
                                collfrt = 1
                                collfree = 0
                                break #check all bbs for collisions again
                                #now that I have adjusted the  new bb
                        else: #no relevant collisions occur. Test next bb
                            continue
                    else: #no relevant collisions occur. Test next bb
                        continue
                else: #no relevant collisions occur. Test next bb
                    continue
            #endfor
        #endwhile
    #endif
    
    return [z,y,x]

def findptsincentervicinity(z,y,x, center, distance, idcs=0):
    '''
    Receives three point cloud vectors (nx1) as input, as well as a center
    point in [z y x] coordinates and a 'distance' parameter.
    
    Returns all points of the point cloud that are <=np.abs(center-distance)
    in an array. I.e. all points in the box around the 'center', where the box
    is defined by 'distance'. Array is of shape n by 3.
    
    If idcs=1, return bbelemsindices, too.
    '''
    #find all points between zmin,zmax , ymin,ymax , xmin,xmax resp.
    
    #not sure if this is a smart and efficient way to do it or not :D
    #it should certainly be faster than for loops through voxels and if
    #cases.
    zpts = (z >= center[0]-(distance//2)+1) * (z <= center[0]+(distance//2))
    ypts = (y >= center[1]-(distance//2)+1) * (y <= center[1]+(distance//2))
    xpts = (x >= center[2]-(distance//2)+1) * (x <= center[2]+(distance//2))

    #find all points that satisfy these conditions at once
    bbelems = zpts*ypts*xpts
    
    #find all of the indices where these conditions are satisfied
    bbelemsindices = np.where(bbelems==True)
    
    bbpts = np.array([z[bbelemsindices], y[bbelemsindices], 
                      x[bbelemsindices]]).transpose()
    #in bbpts are all the points between zmin,zmax , ymin,ymax ,
    #xmin,xmax resp. .
    
    if idcs==0:
        return bbpts
    elif idcs==1:
        return bbpts, bbelemsindices
    
def spbs_searchanddestroyptsinepq(epqz, epqy, epqx, bbpts):
    '''
    Receives the three coordinates of the epq separately as well as all the
    points that are in the current boundary box with center actcntr and 
    size bbsize.
    It then finds all the indices of the points in the epq and then builds new
    lists epqz, epqy, epqx without these points and returns them separately.
    '''
    zmax = np.max(bbpts[:,0])
    zmin = np.min(bbpts[:,0])
    zpts = []
    
    #find all indices that might contain the points of bbpts, we use the lex
    #order in epqz here
    for i in range(0,len(epqz)):
        if ((int(epqz[i])>=int(zmin)) & (int(epqz[i])<=int(zmax))):
            zpts.append(i)
        elif epqz[i]>zmax:
            break
    
    ymax = np.max(bbpts[:,1])
    ymin = np.min(bbpts[:,1])
    ypts = []
    
    #now do the same for all y coordinates, but on the smaller list zpts
    #we don't have a proper "global" lex order here, so we have to check 
    #all elements in zpts
    for ii in zpts:
        if ((int(epqy[ii])>=int(ymin)) & (int(epqy[ii])<=int(ymax))):
            ypts.append(ii)
            
    #now the same procedure for x coordinates and xpts. The indices we get in
    #xpts are the indices that fulfill our requirements on all epqz,y,x
    xmax = np.max(bbpts[:,2])
    xmin = np.min(bbpts[:,2])
    xpts = []
    
    #now do the same for all y coordinates, but on the smaller list zpts
    #we don't have a proper "global" lex order here, so we have to check 
    #all elements in zpts
    for iii in ypts:
        if ((int(epqx[iii])>=int(xmin)) & (int(epqx[iii])<=int(xmax))):
            xpts.append(iii)
    
    #now we lex order the indices in xpts, s.t. we can start deleting the
    #elements in epqz,y,x starting with the highest indices. This way,
    #the removal of elements does not change the indices we found.
    xpts.sort()
    xpts.reverse() #reverse list, so the highest values are first
    
    #remove the points from epq
    for item in xpts:
        epqz.pop(item)
        epqy.pop(item)
        epqx.pop(item)
        
    return epqz, epqy, epqx

def spbs_pointcloud(inp, lex=0, kr=5, rg=1, ocg="str", taulow=0, debug=0):
    """
    Splits the boundary in the input, given as a pointcloud, into
    smaller boundary regions. The size of these regions is specified by kr.
    Use a plane sweep algorithm to do the splitting.
    Returns list of boundary boxes with their centers.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         or four coordinates x y z or x y z intensity or a string that leads 
         to the file that is in x y z or x y z intensity format with no
         header. 
        -'lex' specifies wether the point cloud is already in lexicographic
         order(==1) or not (==0) and if there are duplicate points in the
         point cloud or not (==0, means there might be duplicate values,
         otherwise it is assumed that there aren't).
        -'kr' is the kernel radius.
        -'rg' is the parameter that specifies wether the input grid is
         regular or not and has reasonable values (e.g. not 1.1, but 110).
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the
         usage of a relaxed occupancy grid.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Returns a list of boxes of the same size, that were used to
    split the boundary from one big box into smaller boxes and the 
    center points of the respective boundary boxes. These boundary boxes are
    the strict (0/1) unfilled occupancy grid of the input shape.
    I.e. output is 'out' and 
    out[0]=[boundarybox_0, z,y,x coords of respective center].
    
    There is no default pointcloud, it is assumed the point cloud is not in
    lexicographic order, the default kernel radius is 3 and the debug mode is
    turned off by default.
    
    There's currently a bug with there being too many overlapping boundary 
    boxes. I suspect two areas where there might be an error: 1) regional sls
    is not updated properly, or 2) the collisions are not properly detected.
    """
    if debug==1:
        starttime = time.time()
        
    if ocg=="str":
        x,y,z = spbs_handleinput_pointcloud(inp=inp, lex=lex, rg=rg, ocg="str",
                                            debug=0)
    else:
        x,y,z, vals = spbs_handleinput_pointcloud(inp=inp, lex=lex, rg=rg, 
                                                  ocg="rlx", debug=0)
                        
    if debug==1:
        if ocg=="str":
            print("Initialised the input point cloud (z,y,x) in lex. order.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
            print("Number of Points in the pointcloud: " + str(np.shape(x)[0]))
        else:
            print("Init. the input point cloud (z,y,x,vals) in lex.order.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
            print("Number of Points in the pointcloud: " + str(np.shape(x)[0]))
    
    #get the size of the boundary boxes, i.e. round 2.28*kr up to the next 
    #power. See Pottmann09 for details
    nthpower = int(ceil(log(2.28*kr) / log(2)))
    #thanks SO for the above, see https://stackoverflow.com/questions/4398711
    bbsize = 2**nthpower
    
    #initialize the index of the center point of the bb
    bbcntr = bbsize//2 -1
    
    #shift xrs,yrs,zrs with boundary space in mind
    x = x + (bbsize//2)
    y = y + (bbsize//2)
    z = z + (bbsize//2)
    
    if ocg=="rlx":
        #first, shift values if they are negative, then normalize vals s.t. max is
        #1, then multiply all values such that the maximum is sc
        valsmin = np.min(vals)
        if valsmin<0:
            vals = vals-valsmin #shift to 0
        valsmax = np.max(vals)
        if valsmax==0:
            raise NameError('Maximum Value of the point cloud is 0. Cant \
                            divide by zero')
        vals = (vals/valsmax)
    
    '''
    As all powers of 2 are even, the size of our boundary boxes are even. So,
    compared to our (usual) kernels, there is no unique center point as given 
    a square of width x,the center point would be at (width/2-1/2,width/2-1/2), 
    which is not an integer. A convention is needed. 
    The center point shall be the point (width//2-1,width//2-1), i.e. width//2
    is rounded down to the nearest integer and subtracted by one.
    Examples where the center is marked with *: 
        -width = 8
              [[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, *, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]]
        -width = 4
              [[0, 0, 0, 0],
               [0, *, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]     
        -width = 2
              [[*, 0],
               [0, 0]] 
              
              
    Now, my algorithm, that is inspired by the sweep line algorithm used to
    find line intersections in the book "Computational Geometry" by Berg, 
    Cheong, Kreveld, Overmars from 2008 (Springer), can begin with the basics.
    
    We need more conventions to handle special cases.
    In 2D, the first point that would be chosen as a center point would be
    (ymax, xmax), i.e. the rightmost, topmost point of the point cloud.
    In 3D it's (zmax, ymax, xmax). (order is important)
    
    Imagine the sweep line (in 2D) or sweep plane (in 3D) coming in from the 
    top/front (y/z-axis positive, moving to 0).
    Once it hits one or multiple points, it chooses a point according to the 
    priorities above, i.e. highest z value before highest y value before 
    highest x value. This also means, that there is no need for an event point
    queue like in the original algorithm, because the points that would be
    in the event point queue are all in our input list, in the right order.
    The original algorithm was checking for intersection points between the
    line segments, which had to be added to the event point queue. This is
    obviously not necessary in our case.
    
    But, the system with deleting unnecessary points from the event queue
    still holds up and doing that with the input list would basically make it
    unusable for anything else. Especially, border cases, where multiple
    boundary boxes meet would be much harder to handle. That's why, I decided
    to copy the input list, reverse it, and make it my event point queue. 
    
    When a point from the event point queue (epq) is added to a boundary box,
    it is deleted from the epq. 
    '''
    
    #initialize global sweep plane status and regional sweep plane status
    gsps = []
    rsps = []
    '''
    The global sweep line/plane status is a list of all boundary boxes that the
    sweep line is currently hitting. 
    
    Why do we need the sls in the first place? It stores all boundary boxes,
    that might cause collision issues when initalizing a new boundary box.
    To save time, we're not checking the collisions of a new boundary box with
    all boundary boxes, just the ones it might actually be able to hit.
    
    To improve on this, we induce a regional sps, that contains only the bbs
    from the gsps that are even close to the current bb.
    
    The elements of the gsps,rsps consist of the coordinates [z,y,x] of the
    actual center of the respective boundary boxes.
    '''
    
    #initialize output list (contains volume of boundary box + actual center
    #coordinate)
    out = []
    
    #initialize event point queue(s). Instead of using one epq, we use one for
    #each coordinate. This is just a matter of personal taste and can be done
    #differently. One could also use an array as epq, and set the handled
    #events to a value that can be easily recognized, s.t. handled events
    #can be skipped
    epqx = x.tolist()
    epqy = y.tolist()
    epqz = z.tolist()
    
    if debug==1:
        print("Everything is initialised. Next stop, main while loop.\n"
              + "Current Runtime: " + str(time.time() - starttime))
    
    #main loop of boundary box construction and event handling
    while(epqx): #when all event points are handled the event
    #point queue is empty and we are done. This is the loop where we handle
    #all these event points.
        '''
        What's happening here?
        1) We start by choosing our next potential center point of 
        the boundary box that we want to add. This point is called 
        'cntr' and is always the next non-(-1,-1,-1)-point in the event point
        queue 'epq'. 
        2) Now we have to update our sweep plane, i.e. sweep plane
        status 'sps' such that all active boundary boxes are
        contained in it. 
        A boundary box is active in the current
        sweep plane (which position we get by considering the 
        coordinates of the current point 'cntr') if 
        cntr_z+4>zmin_bbs (i.e. zmaxnew>zmin_bbs). Otherwise remove
        the boundary box from the sweep plane status.
        3) With our current point and an up-to-date sweep line
        status, we can go ahead and check the new boundary box
        around 'cntr' for collisions with other boxes and 
        adjust/shift the box accordingly. Through this shifting,
        the actual center point of the boundary may no longer be
        'cntr'.
        4) Now that we have our new boundary box (without collisions
        as far as possible), we create an occupancy grid of the shape
        in our boundary box and save that to 'bbvolume', delete
        all points of the input data that we hit with our boundary
        box from the epq, add the 'new' boundary box to the sweep line 
        status and add the new bbvolume to the output together with the
        accurate center point coordinates of the current bbvolume.
        '''
        #next "center" pt of bb i.e. local zmax, ymax,
        #xmax. Contains the coords of cntr in 3D space
        #eqp is lex ordered, highest values come last
        epqlen = len(epqz)
        cntr=np.array([epqz[epqlen-1],epqy[epqlen-1],epqx[epqlen-1]]) 
        
        if debug ==1:
            print("cntr "+str(cntr))
        
        if debug==1:
            print("Cntr init complete.\n"
                  +"Length of gsps before "+str(len(gsps))+".\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        rsps = [] #reset regional sweep plane status after each event point
        
        if gsps: #if sls not empty, update sweep line status
            #this for loop might look at the same box twice and therefore
            #might duplicate boxes in rsps
            for j in range(len(gsps)-1,-1,-1): #actbb for "active boundary box"
                if (cntr[0] - gsps[j][0] <= -bbsize): #i.e. when there's no
                                                    #no collision on z-axis
                                                    #remove point from gsps
                    gsps.pop(j)
                elif (((cntr[0] - gsps[j][0] > -bbsize)
                       & (cntr[0] - gsps[j][0] <= -(bbsize//2)))#POTfrontcollision
                      & (np.abs(cntr[1] - gsps[j][1]) <= bbsize)#POTtopORbottcoll
                      & (np.abs(cntr[2] - gsps[j][2]) <= bbsize)):#POTrightORleftcoll
                #these collision points correspond to the neighboring bbs with
                #the new bb as baseline. I.e., "potrightcollision" means the
                #right side of the new bb has Maybe a collision with the left
                #side of the neighboring bb
                    rsps = rsps + [[gsps[j][0],gsps[j][1],gsps[j][2]]]
                
        #remove duplicates from rsps by inducing lex order and then checking
        #for dupes
        rsps = sorted(rsps)
        for j in range(len(rsps)-1,0,-1):
            if rsps[j]==rsps[j-1]:
                rsps.pop(j)
        
        #initialize/reset the current boundary box volume
        if ocg=="rlx":
            bbvolume = np.zeros((bbsize,bbsize,bbsize),np.double) 
        else:
            bbvolume = np.zeros((bbsize,bbsize,bbsize),int) 
        
        #check for collisions
        actcntr = spbs_checkboxcollisions(newboxcntr=cntr, rsps=rsps,
                                          bbsize=bbsize)
        # print(actcntr)
        # print(bbsize)
        # print("Length of rsps "+str(len(rsps)))
        #if debug==1:
            # if len(rsps)>0:
            #     print("RSPS >0 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        '''
        Occgrid construction, saving bbvolume, updating sls, 5 steps:
            1) search all pts inside the bb and get the coordinates.
            2) By calculating the distance between the actual center of the bb
               and the pts in bbpts, find out where in bbvolume the pts are and
               give these pts the value 1 in bbvolume.
            3) Find all bbpts in the epq and remove them from it. Because 
               they are already in a boundary box, they have been handled. I.e. 
               they no longer have to be considered as centers of a bb in 
               the future. 
            4) Save bbvolume and the actual center point actctnr to 'out'.
            5) Update sls, i.e. add current bb to sls.
                                                  
        '''
        #step 1)
        #find all points between zmin,zmax , ymin,ymax , xmin,xmax resp.
        if ocg=="str":
            bbpts = findptsincentervicinity(z=z,y=y,x=x, center=actcntr,
                                            distance=bbsize)
            #in bbpts are all the points between zmin,zmax , ymin,ymax ,
            #xmin,xmax resp. . So, Step1 is complete.
        else:
            bbpts, bbidcs = findptsincentervicinity(z=z,y=y,x=x, 
                                                    center=actcntr,
                                                    distance=bbsize, idcs=1)
            bbvals = np.transpose([vals[bbidcs],1])
        
        #step 2)
        #populate bbvolume
        if ocg=="str":
            for ii in range(0,np.shape(bbpts)[0]):
                ztemp, ytemp, xtemp = bbpts[ii]
                zdist = actcntr[0] - ztemp
                ydist = actcntr[1] - ytemp
                xdist = actcntr[2] - xtemp
    
                bbvolume[bbcntr - zdist, bbcntr - ydist, bbcntr - xdist] = 1
        else:
            for ii in range(0,np.shape(bbpts)[0]):
                if bbvals[ii]>taulow:
                    ztemp, ytemp, xtemp = bbpts[ii]
                    zdist = actcntr[0] - ztemp
                    ydist = actcntr[1] - ytemp
                    xdist = actcntr[2] - xtemp
                
                    bbvolume[bbcntr - zdist, bbcntr - ydist,
                             bbcntr - xdist] = bbvals[ii]
            
        #step 3)
        #Now we populated bbvolume. Next, find all bbpts in the epq and 
        #delete them from it.
        
        #if np.max(bbvolume)!=0: #i.e. do this if bbpts not empty
        epqz, epqy, epqx = spbs_searchanddestroyptsinepq(epqz=epqz,
                                                         epqy=epqy,
                                                         epqx=epqx, 
                                                         bbpts=bbpts)
        #else: if this case even happens, we got a problem. bbpts shouldn't
        #be empty
        
        
        
        #step 4)
        #Our epq is up-to-date and our bbvolume is populated. Now we can 
        #save our results (the new bb and its actual center) to the output
        #list
        actcntrarr = np.asarray(actcntr)
        out.append([bbvolume,actcntrarr])
        #we will need the actcntr for the consistent filling. That's why 
        #it's included. Actcntr contains the coordinates (!) of the actual
        #center.
        
        #step 5)
        #we added the new bb to the output, we handled all cases, what is 
        #left is to add the boundary values of the new boundary box to the 
        #sweepline/plane status, so we can check it for collisions in the
        #future.
        #what's left is to update the gsps and add the new box to it.
        
        gsps.append(actcntr)
                
        if debug==1:
            print("Length of x "+str(len(x))+".\n"
                  +"Length of Epq "+str(len(epqz))+".\n"
                  +"Length of gsps after "+str(len(gsps))+".\n"
                  +"Length of rsps "+str(len(rsps))+".\n"
                  +"Length of bbpts "+str(len(bbpts))+".\n"
                  +"cntr "+str(cntr)+".\n"
                  +"actcntr "+str(actcntr)+".\n"
                  + "Current Runtime: " + str(time.time()-starttime)+".\n"
                  + "####################")
    return out

def spbs_ocg(inp, kr=5, ocg="str", taulow=0, debug=0):
    """
    Splits the boundary in the input, given as an occupancy grid, into
    smaller boundary regions. The size of these regions is specified by kr.
    Use a plane sweep algorithm to do the splitting.
    Returns list of boundary boxes with their centers.
    
    Input:
        -'inp' is a strict or relaxed occupancy grid
        -'kr' is the kernel radius.
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the
         usage of a relaxed occupancy grid.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Returns a list of boxes of the same size, that were used to
    split the boundary from one big box into smaller boxes and the 
    center points of the respective boundary boxes. These boundary boxes are
    the strict (0/1) unfilled occupancy grid of the input shape.
    I.e. output is 'out' and 
    out[0]=[boundarybox_0, z,y,x coords of respective center].
    
    There is no default pointcloud, it is assumed the point cloud is not in
    lexicographic order, the default kernel radius is 3 and the debug mode is
    turned off by default.
    """
    if debug==1:
        starttime = time.time()
    
    if ocg=="str":
        x,y,z = spbs_handleinput_ocg(inp=inp, ocg=0)
    else:
        x,y,z,vals = spbs_handleinput_ocg(inp=inp, ocg=1)
                        
    if debug==1:
        print("Initialised the input point cloud (z,y,x) in lex. order.\n"
              + "Current Runtime: " + str(time.time() - starttime))
        print("Number of Points in the pointcloud: " + str(np.shape(x)[0]))
    
    #get the size of the boundary boxes, i.e. round 2.28*kr up to the next 
    #power. See Pottmann09 for details
    nthpower = int(ceil(log(2.28*kr) / log(2)))
    #thanks SO for the above, see https://stackoverflow.com/questions/4398711
    bbsize = 2**nthpower
    
    #initialize the index of the center point of the bb
    bbcntr = bbsize//2 -1
    
    #shift xrs,yrs,zrs with boundary space in mind
    x = x + (bbsize//2)
    y = y + (bbsize//2)
    z = z + (bbsize//2)
    
    if ocg=="rlx":
        #first, shift values if they are negative, then normalize vals s.t. max is
        #1, then multiply all values such that the maximum is sc
        valsmin = np.min(vals)
        if valsmin<0:
            vals = vals-valsmin #shift to 0
        valsmax = np.max(vals)
        if valsmax==0:
            raise NameError('Maximum Value of the point cloud is 0. Cant \
                            divide by zero')
        vals = (vals/valsmax)
        
    '''
    As all powers of 2 are even, the size of our boundary boxes are even. So,
    compared to our (usual) kernels, there is no unique center point as given 
    a square of width x,the center point would be at (width/2-1/2,width/2-1/2), 
    which is not an integer. A convention is needed. 
    The center point shall be the point (width//2-1,width//2-1), i.e. width//2
    is rounded down to the nearest integer and subtracted by one.
    Examples where the center is marked with *: 
        -width = 8
              [[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, *, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]]
        -width = 4
              [[0, 0, 0, 0],
               [0, *, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]     
        -width = 2
              [[*, 0],
               [0, 0]] 
              
              
    Now, my algorithm, that is inspired by the sweep line algorithm used to
    find line intersections in the book "Computational Geometry" by Berg, 
    Cheong, Kreveld, Overmars from 2008 (Springer), can begin with the basics.
    
    We need more conventions to handle special cases.
    In 2D, the first point that would be chosen as a center point would be
    (ymax, xmax), i.e. the rightmost, topmost point of the point cloud.
    In 3D it's (zmax, ymax, xmax). (order is important)
    
    Imagine the sweep line (in 2D) or sweep plane (in 3D) coming in from the 
    top/front (y/z-axis positive, moving to 0).
    Once it hits one or multiple points, it chooses a point according to the 
    priorities above, i.e. highest z value before highest y value before 
    highest x value. This also means, that there is no need for an event point
    queue like in the original algorithm, because the points that would be
    in the event point queue are all in our input list, in the right order.
    The original algorithm was checking for intersection points between the
    line segments, which had to be added to the event point queue. This is
    obviously not necessary in our case.
    
    But, the system with deleting unnecessary points from the event queue
    still holds up and doing that with the input list would basically make it
    unusable for anything else. Especially, border cases, where multiple
    boundary boxes meet would be much harder to handle. That's why, I decided
    to copy the input list, reverse it, and make it my event point queue. 
    
    When a point from the event point queue (epq) is added to a boundary box,
    it is deleted from the epq. 
    '''
    
    #initialize global sweep plane status and regional sweep plane status
    gsps = []
    #rsps = []
    '''
    The global sweep line/plane status is a list of all boundary boxes that the
    sweep line is currently hitting. 
    
    Why do we need the sls in the first place? It stores all boundary boxes,
    that might cause collision issues when initalizing a new boundary box.
    To save time, we're not checking the collisions of a new boundary box with
    all boundary boxes, just the ones it might actually be able to hit.
    
    To improve on this, we induce a regional sps, that contains only the bbs
    from the gsps that are even close to the current bb.
    
    The elements of the gsps,rsps consist of the coordinates [z,y,x] of the
    actual center of the respective boundary boxes.
    '''
    
    #initialize output list (contains volume of boundary box + actual center
    #coordinate)
    out = []
    
    #initialize event point queue(s). Instead of using one epq, we use one for
    #each coordinate. This is just a matter of personal taste and can be done
    #differently. One could also use an array as epq, and set the handled
    #events to a value that can be easily recognized, s.t. handled events
    #can be skipped
    epqx = x.tolist()
    epqy = y.tolist()
    epqz = z.tolist()
    
    if debug==1:
        print("Everything is initialised. Next stop, main while loop.\n"
              + "Current Runtime: " + str(time.time() - starttime))
    
    #main loop of boundary box construction and event handling
    while(epqx): #when all event points are handled the event
    #point queue is empty and we are done. This is the loop where we handle
    #all these event points.
        '''
        What's happening here?
        1) We start by choosing our next potential center point of 
        the boundary box that we want to add. This point is called 
        'cntr' and is always first element in the event point
        queue 'epq'. 
        2) Now we have to update our sweep plane, i.e. sweep plane
        status 'sps' such that all active boundary boxes are
        contained in it. 
        A boundary box is active in the current
        sweep plane (which position we get by considering the 
        coordinates of the current point 'cntr') if 
        cntr_z+bbsize//2>zmin_bbs (i.e. zmaxnew>zmin_bbs). Otherwise remove
        the boundary box from the sweep plane status.
        3) With our current point and an up-to-date sweep line
        status, we can go ahead and check the new boundary box
        around 'cntr' for collisions with other boxes and 
        adjust/shift the box accordingly. Through this shifting,
        the actual center point of the boundary may no longer be
        'cntr'.
        4) Now that we have our new boundary box (without collisions
        as far as possible), we create an occupancy grid of the shape
        in our boundary box and save that to 'bbvolume', delete
        all points of the input data that we hit with our boundary
        box from the epq, add the 'new' boundary box to the sweep line 
        status and add the new bbvolume to the output together with the
        accurate center point coordinates of the current bbvolume.
        '''
        #next "center" pt of bb i.e. local zmax, ymax,
        #xmax. Contains the coords of cntr in 3D space
        #eqp is lex ordered, highest values come last
        epqlen = len(epqz)
        cntr=np.array([epqz[epqlen-1],epqy[epqlen-1],epqx[epqlen-1]]) 
        
        if debug ==1:
            print("cntr "+str(cntr))
        
        if debug==1:
            print("Cntr init complete.\n"
                  +"Length of gsps before "+str(len(gsps))+".\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        rsps = [] #reset regional sweep plane status after each event point
        
        if gsps: #if sls not empty, update sweep line status
            #this for loop might look at the same box twice and therefore
            #might duplicate boxes in rsps
            for j in range(len(gsps)-1,-1,-1): #actbb for "active boundary box"
                if (cntr[0] - gsps[j][0] <= -bbsize): #i.e. when there's no
                                                    #no collision on z-axis
                                                    #remove point from gsps
                    gsps.pop(j)
                elif (((cntr[0] - gsps[j][0] > -bbsize)
                       & (cntr[0] - gsps[j][0] <= -(bbsize//2)))#POTfrontcollision
                      & (np.abs(cntr[1] - gsps[j][1]) <= bbsize)#POTtopORbottcoll
                      & (np.abs(cntr[2] - gsps[j][2]) <= bbsize)):#POTrightORleftcoll
                #these collision points correspond to the neighboring bbs with
                #the new bb as baseline. I.e., "potrightcollision" means the
                #right side of the new bb has Maybe a collision with the left
                #side of the neighboring bb
                    rsps = rsps + [[gsps[j][0],gsps[j][1],gsps[j][2]]]
                
        #remove duplicates from rsps by inducing lex order and then checking
        #for dupes
        rsps = sorted(rsps)
        for j in range(len(rsps)-1,0,-1):
            if rsps[j]==rsps[j-1]:
                rsps.pop(j)
        
        #initialize/reset the current boundary box volume
        if ocg=="rlx":
            bbvolume = np.zeros((bbsize,bbsize,bbsize),np.double) 
        else:
            bbvolume = np.zeros((bbsize,bbsize,bbsize),int) 
        
        #check for collisions
        actcntr = spbs_checkboxcollisions(newboxcntr=cntr, rsps=rsps,
                                          bbsize=bbsize)
        # print(actcntr)
        # print(bbsize)
        # print("Length of rsps "+str(len(rsps)))
        #if debug==1:
            # if len(rsps)>0:
            #     print("RSPS >0 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        '''
        Occgrid construction, saving bbvolume, updating sls, 5 steps:
            1) search all pts inside the bb and get the coordinates.
            2) By calculating the distance between the actual center of the bb
               and the pts in bbpts, find out where in bbvolume the pts are and
               give these pts the value 1 in bbvolume.
            3) Find all bbpts in the epq and remove them from it. Because 
               they are already in a boundary box, they have been handled. I.e. 
               they no longer have to be considered as centers of a bb in 
               the future. 
            4) Save bbvolume and the actual center point actctnr to 'out'.
            5) Update sls, i.e. add current bb to sls.
                                                  
        '''
        #step 1)
        #find all points between zmin,zmax , ymin,ymax , xmin,xmax resp.
        if ocg=="str":
            bbpts = findptsincentervicinity(z=z,y=y,x=x, center=actcntr,
                                            distance=bbsize)
            #in bbpts are all the points between zmin,zmax , ymin,ymax ,
            #xmin,xmax resp. . So, Step1 is complete.
        else:
            bbpts, bbidcs = findptsincentervicinity(z=z,y=y,x=x, 
                                                    center=actcntr,
                                                    distance=bbsize, idcs=1)
            bbvals = np.transpose([vals[bbidcs],1])
        
        #step 2)
        #populate bbvolume
        if ocg=="str":
            for ii in range(0,np.shape(bbpts)[0]):
                ztemp, ytemp, xtemp = bbpts[ii]
                zdist = actcntr[0] - ztemp
                ydist = actcntr[1] - ytemp
                xdist = actcntr[2] - xtemp
    
                bbvolume[bbcntr - zdist, bbcntr - ydist, bbcntr - xdist] = 1
        else:
            for ii in range(0,np.shape(bbpts)[0]):
                if bbvals[ii]>taulow:
                    ztemp, ytemp, xtemp = bbpts[ii]
                    zdist = actcntr[0] - ztemp
                    ydist = actcntr[1] - ytemp
                    xdist = actcntr[2] - xtemp
                
                    bbvolume[bbcntr - zdist, bbcntr - ydist,
                             bbcntr - xdist] = bbvals[ii]
            
        #step 3)
        #Now we populated bbvolume. Next, find all bbpts in the epq and 
        #delete them from it.
        
        #if np.max(bbvolume)!=0: #i.e. do this if bbpts not empty
        epqz, epqy, epqx = spbs_searchanddestroyptsinepq(epqz=epqz,
                                                         epqy=epqy,
                                                         epqx=epqx, 
                                                         bbpts=bbpts)
        #else: if this case even happens, we got a problem. bbpts should never
        #be empty

        
        
        
        #step 4)
        #Our epq is up-to-date and our bbvolume is populated. Now we can 
        #save our results (the new bb and its actual center) to the output
        #list
        actcntrarr = np.asarray(actcntr)
        out.append([bbvolume,actcntrarr])
        #we will need the actcntr for the consistent filling. That's why 
        #it's included. Actcntr contains the coordinates (!) of the actual
        #center.
        
        #step 5)
        #we added the new bb to the output, we handled all cases, what is 
        #leftis to add the boundary values of the new boundary box to the 
        #sweepline/plane status, so we can check it for collisions in the
        #future.
        #what's left is to update the gsps and add the new box to it.
        
        gsps.append(actcntr)
                
        if debug==1:
            print("Length of x "+str(len(x))+".\n"
                  +"Length of Epq "+str(len(epqz))+".\n"
                  +"Length of gsps after "+str(len(gsps))+".\n"
                  +"Length of rsps "+str(len(rsps))+".\n"
                  +"Length of bbpts "+str(len(bbpts))+".\n"
                  +"cntr "+str(cntr)+".\n"
                  +"actcntr "+str(actcntr)+".\n"
                  + "Current Runtime: " + str(time.time()-starttime)+".\n"
                  + "####################")
    return out

def simplebs_pointcloud(inp, kr=3, rg=1, ocg="str", taulow=0, debug=0):
    """
    Splits the boundary in the input, given as a pointcloud, into
    smaller boundary regions. The size of these regions is specified by kr.
    Does the splitting by enforcing a grid of a certain size onto the object.
    Returns list of boundary boxes with their centers.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         or four coordinates x y z or x y z intensity or a string that leads 
         to the file that is in x y z or x y z intensity format with no
         header. 
        -'kr' is the kernel radius.
        -'rg' is the parameter that specifies wether the input grid is
         regular or not and has reasonable values (e.g. not 1.1, but 110).
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the
         usage of a relaxed occupancy grid.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Returns a list of boxes of one same size, that were used to
    split the boundary from one big box into smaller boxes and the 
    center points of the respective boundary boxes. These boundary boxes are
    the strict (0/1) unfilled occupancy grid of the input shape.
    I.e. output is 'out' and 
    out[0]=[boundarybox_0, z,y,x coords of respective center].
    
    
    There is no default pointcloud, it is assumed the point cloud is not in
    lexicographic order, the default kernel radius is 3 and the debug mode is
    turned off by default.
    """
    if debug==1:
        starttime = time.time()
        
    if rg!=1:
        if len(rg)!=2:
            raise NameError('Parameter rg specifies wether the input is on a \
                            regular grid or not, and if it has reasonable \
                            values or not. If rg==1, the input is supposed \
                            to have a regular grid and reasonable values. \
                            In the other case, we have to deal with special \
                            circumstances, which require a rescaling \
                            parameter rho (in the same way as it was used \
                            in the old occgrid method). I.e. the input shall \
                            be a [0,rho]. Example: If xmax is 0.9, choosing \
                            rho=10 or 100 should give good results.')
        else:
            rho = rg[1]
    else:
        rho = 1 #i.e. there won't be any rescaling


    if type(inp)==str:
        #load pointcloud
        if ocg=="str":
            x, y, z = np.loadtxt(inp,skiprows=1, unpack=True,
                             usecols=(0,1,2))
        else:
            x, y, z, vals = np.loadtxt(inp,skiprows=1, unpack=True,
                             usecols=(0,1,2,3))
        
        #find min,max values of each axis
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        zmin = np.min(z)
        zmax = np.max(z)
        
        #normalise x,rescale axes to max val (both if irreg grid) and conform
        #to regular grid values by making it int. This also has the nice
        #effect of reducing the volume to its actual values(i.e. no borders of 
        #zeros or sth)
        xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
        yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
        zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)
        
    elif isinstance(inp,np.ndarray):
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]
        
        if ocg=="rlx":
            vals = inp[:,3]
        
        #find min,max values of each axis
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        zmin = np.min(z)
        zmax = np.max(z)
        
        #normalise x and then rescale axes to max val
        xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
        yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
        zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)

        
    elif isinstance(inp,list):
        #we assume the input list is given in
        #z y x format
         
        inp = np.asarray(inp)
            
        #get the separate coordinate vectors
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]
        
        if ocg=="rlx":
            vals = inp[:,3]
        
        #find min,max values of each axis
        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        zmin = np.min(z)
        zmax = np.max(z)
        
        #normalise x and then rescale axes to max val
        xrs = (((x-xmin)/(xmax-xmin))*((xmax-xmin)*rho)).astype(int)
        yrs=  (((y-ymin)/(ymax-ymin))*((ymax-ymin)*rho)).astype(int)
        zrs = (((z-zmin)/(zmax-zmin))*((zmax-zmin)*rho)).astype(int)

    else:
        raise NameError('Input can be an already loaded pointcloud that \
                        consists of the three coordinates z y x or a string \
                        that leads to the file that is in z y x format with \
                        no header.')
                        
    if debug==1:
        print("Initialised the input point cloud (z,y,x) in lex. order.\n"
              + "Current Runtime: " + str(time.time() - starttime))
        print("Number of Points in the pointcloud: " + str(np.shape(x)[0]))
    
    #get the size of the boundary boxes, i.e. round 2.28*kr up to the next 
    #power. See Pottmann09 for details
    nthpower = int(ceil(log(2.28*kr) / log(2)))
    #thanks SO for the above, see https://stackoverflow.com/questions/4398711
    bbsize = 2**nthpower
    
    #initialize the index of the center point of the bb
    bbcntr = bbsize//2 -1
    
    '''
    As all powers of 2 are even, the size of our boundary boxes are even. So,
    compared to our (usual) kernels, there is no unique center point as given 
    a square of width x,the center point would be at (width/2-1/2,width/2-1/2), 
    which is not an integer. A convention is needed. 
    The center point shall be the point (width//2-1,width//2-1), i.e. width//2
    is rounded down to the nearest integer and subtracted by one.
    Examples where the center is marked with *: 
        -width = 8
              [[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, *, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]]
        -width = 4
              [[0, 0, 0, 0],
               [0, *, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]     
        -width = 2
              [[*, 0],
               [0, 0]] 
    '''
    #initialize output list (contains volume of boundary box + actual center
    #coordinate)
    out = []
    
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
    
    #find min values of each axis
    zmin = np.min(zrs) - bbsize//2 +1
    ymin = np.min(yrs) - bbsize//2 +1
    xmin = np.min(xrs) - bbsize//2 +1
    
    #shift values s.t. the minimums are at 0.
    zrs = zrs - zmin
    yrs = yrs - ymin
    xrs = xrs - xmin
    
    #find max vals of each axis
    zmax = np.max(zrs) + bbsize//2
    ymax = np.max(yrs) + bbsize//2
    xmax = np.max(xrs) + bbsize//2
    
    #get the amount of boundary boxes that are needed on each axis, if there
    #were points everywhere in the above volume from zmin2zmax,ymin2ymax,
    #xmin2max
    if zmax%bbsize==0:
        amtbbsz = zmax//bbsize #'amtbbsz' for amount of boundary boxes on z
    else:
        zmax = zmax + (bbsize - (zmax % bbsize))#extend zmax such that the 
        #last boundary box fits, keep in mind that zmax is just a substitute
        #for the highest coordinate, not the coordinate directly.
        amtbbsz = zmax//bbsize# +1
        # #add zero padding
        # if pad==0:
        #     zrs = np.pad(zrs, (0,bbsize-zmax%bbsize), 'constant')
        # elif pad==1:
        #     zrs = np.pad(zrs, (0,bbsize-zmax%bbsize), 'reflect') 
        # else:
        #     zrs = np.pad(zrs, (0,bbsize-zmax%bbsize), 'edge') 

    if ymax%bbsize==0:
        amtbbsy = ymax//bbsize
    else:
        ymax = ymax + (bbsize - (ymax % bbsize))
        amtbbsy = ymax//bbsize# +1
        
    if xmax%bbsize==0:
        amtbbsx = xmax//bbsize
    else:
        xmax = xmax + (bbsize - (xmax % bbsize))
        amtbbsx = xmax//bbsize# +1

    #create a 2D array that contains the center points of all potential
    #boundary boxes. Each line represents a boundary box.
    #The first three entries are the z y x coordinates
    #(, then a boolean value. This boolean decides wether a boundary box 
    #contains values or not.)<-- deprecated, not in use any longer
    bbcntrs = np.zeros([amtbbsx*amtbbsy*amtbbsz,3],int)
    
    counter = 0
    #now fill the 2D array with the center coordinates
    for i in range(bbsize//2 -1,zmax,bbsize):
        for ii in range(bbsize//2 -1,ymax,bbsize):
            for iii in range(bbsize//2 -1,xmax,bbsize):
                bbcntrs[counter,:] = [i, ii, iii]
                counter = counter +1
        
    #now, for each individual potential box, we want to find out if and which
    #points are contained in it
    for i in range(0,np.shape(bbcntrs)[0]):
        #find min,max values of for the current bb
        zmaxbb = bbcntrs[i,0] + bbsize//2
        zminbb = bbcntrs[i,0] - bbsize//2 +1
        ymaxbb = bbcntrs[i,1] + bbsize//2
        yminbb = bbcntrs[i,1] - bbsize//2 +1
        xmaxbb = bbcntrs[i,2] + bbsize//2
        xminbb = bbcntrs[i,2] - bbsize//2 +1
        
        #now, find all points that are inside the box
    
        #not sure if this is a smart and efficient way to do it or not :D
        #it should certainly be faster than for loops through voxels and if
        #cases.
        zpts = (zrs>=zminbb) * (zrs<=zmaxbb) #zpts has a 1 on all entries of 
                                             #zrs that are smaller than zmaxbb
                                             #and bigger than zminbb. 
        ypts = (yrs>=yminbb) * (yrs<=ymaxbb)
        xpts = (xrs>=xminbb) * (xrs<=xmaxbb)

        #find all points that satisfy these conditions at once
        bbelems = zpts*ypts*xpts
        
        #find all of the indices where these conditions are satisfied
        bbelemsindices = np.where(bbelems==True)
        
        #only append a boundary box to out, if there are values in bb
        if np.shape(bbelemsindices)[1]!=0 : 
            bbpts = np.array([zrs[bbelemsindices], yrs[bbelemsindices], 
                          xrs[bbelemsindices]]).transpose()
        
            #We have all the values inside the current boundary box via 
            #bbelemsindices. Now we have to put them into the boundary box at 
            #the right positions. Remember bbpts contains the coordinates of 
            #the values we want. We want them at the right position in bbvol
            #and put the value 1 there instead of the coordinates.
            
            if ocg=="str":
                bbvol = np.zeros([bbsize,bbsize,bbsize], int)
            else:
                bbvol = np.zeros([bbsize,bbsize,bbsize], np.double)
                
            #populate bbvolume
            if ocg=="str":
                for ii in range(0,np.shape(bbpts)[0]):
                    #calculate distance between center and current point on each axis
                    zdist = bbcntrs[i,0] - bbpts[ii,0]
                    ydist = bbcntrs[i,1] - bbpts[ii,1]
                    xdist = bbcntrs[i,2] - bbpts[ii,2]
                    bbvol[bbcntr - zdist, bbcntr - ydist, bbcntr - xdist] = 1
            else:
                for ii in range(0,np.shape(bbpts)[0]):
                    if vals[bbelemsindices[0][ii]]>taulow:
                        #calculate distance between center and current point on each axis
                        zdist = bbcntrs[i,0] - bbpts[ii,0]
                        ydist = bbcntrs[i,1] - bbpts[ii,1]
                        xdist = bbcntrs[i,2] - bbpts[ii,2]
                        bbvol[bbcntr - zdist, bbcntr - ydist,
                              bbcntr - xdist] = vals[bbelemsindices[0][ii]]
                
            out.append([bbvol,bbcntrs[i]])

            
    if debug==1:
        print("Amount of bbs in output " + str(len(out)) +"\n"
              + "Done.\n"
              + "Current Runtime: " + str(time.time() - starttime))
        
    return out


def simplebs_ocg(inp, kr=5, pad=0, ocg="str", taulow=0, debug=0):
    """
    Splits the boundary in the input, given as an occupancy grid, into
    smaller boundary regions. The size of these regions is specified by kr.
    Does the splitting by enforcing a grid of a certain size onto the object.
    Returns list of boundary boxes with their centers.
    
    Input:
        -'inp' is a strict or relaxed occupancy grid that was made, for 
         example, with the 'createoccgrid' method in this file. It's an array.
        -'kr' is the kernel radius.
        -'pad' is the parameter that chooses how the padding of the occupancy
         grid is done.If ==0, then just zero-padding is used. If ==1,2 another
         hardcoded mode is used. If pad==1,reflect mode is used for padding, 
         if pad==2 sth else.
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the
         usage of a relaxed occupancy grid.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Returns 'out', 
    a list of boxes of one same size, that were used to
    split the boundary from one big box into smaller boxes and the 
    center points of the respective boundary boxes. These boundary boxes are
    the strict (0/1) unfilled occupancy grid of the (strict) input occupancy 
    grid.
    I.e. output is 'out' and 
    out[0]=[boundarybox_0, z,y,x coords of respective center].
    """
    if debug==1:
        starttime = time.time()

    
    if not isinstance(inp,np.ndarray):
        raise NameError('Input has to be strict occgrid as np array.')
    
    #get the size of the boundary boxes, i.e. round 2.28*kr up to the next 
    #power. See Pottmann09 for details
    nthpower = int(ceil(log(2.28*kr) / log(2)))
    #thanks SO for the above, see https://stackoverflow.com/questions/4398711
    bbsize = 2**nthpower
    
    if debug==1:
        print("Current Runtime: " + str(time.time() - starttime)
              +"\n Number of Voxels in input occgrid: " + str(np.shape(inp)[0]*
                                                          np.shape(inp)[1]*
                                                          np.shape(inp)[2])
              +"\n Size of boundary boxes: " + str(bbsize))
        
    '''
    As all powers of 2 are even, the size of our boundary boxes are even. So,
    compared to our (usual) kernels, there is no unique center point as given 
    a square of width x,the center point would be at (width/2-1/2,width/2-1/2), 
    which is not an integer. A convention is needed. 
    The center point shall be the point (width//2-1,width//2-1), i.e. width//2
    is rounded down to the nearest integer and subtracted by one.
    Examples where the center is marked with *: 
        -width = 8
              [[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, *, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0]]
        -width = 4
              [[0, 0, 0, 0],
               [0, *, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]     
        -width = 2
              [[*, 0],
               [0, 0]] 
    '''
    #initialize output list (contains volume of boundary box + actual center
    #coordinate)
    out = []
    
    if ocg=="rlx":
        #first, shift values if they are negative, then normalize vals s.t. max is
        #1, then multiply all values such that the maximum is sc
        inpmin = np.min(inp)
        if inpmin<0:
            inp = inp-inpmin #shift to 0
        inpmax = np.max(inp)
        if inpmax==0:
            raise NameError('Maximum Value of the point cloud is 0. Cant \
                            divide by zero')
        inp = (inp/inpmax)
        
        thrvals = np.where(inp<=taulow)
        inp[thrvals] = 0
    
    #get the amount of boundary boxes that are needed on each axis, if there
    #were points everywhere in the above volume from zmin2zmax,ymin2ymax,
    #xmin2max (this comment is still in the coordinate notation due to 
    #copy&paste. See boundarysplitsimple for original method.)
    if np.shape(inp)[0]%bbsize==0:
        amtbbsz = np.shape(inp)[0]//bbsize #'amtbbsz' for amount of bbs on z
    else: 
        amtbbsz = np.shape(inp)[0]//bbsize +1
        #add zero padding
        if pad==0:
            inp = np.pad(inp, ((0,bbsize-np.shape(inp)[0]%bbsize), (0,0),
                               (0,0)), 'constant')
        elif pad==1:
            inp = np.pad(inp, ((0,bbsize-np.shape(inp)[0]%bbsize), (0,0),
                               (0,0)), 'reflect') 
        else:
            inp = np.pad(inp, ((0,bbsize-np.shape(inp)[0]%bbsize), (0,0),
                               (0,0)), 'edge') 
            
    if np.shape(inp)[1]%bbsize==0:
        amtbbsy = np.shape(inp)[1]//bbsize
    else:
        #ymax = ymax + (bbsize - (ymax % bbsize)) #same as above.
        amtbbsy = np.shape(inp)[1]//bbsize +1
        #add zero padding
        if pad==0:
            inp = np.pad(inp, ((0,0), (0,bbsize-np.shape(inp)[1]%bbsize),
                               (0,0)),'constant')
        elif pad==1:
            inp = np.pad(inp, ((0,0), (0,bbsize-np.shape(inp)[1]%bbsize),
                               (0,0)), 'reflect')
        else:
            inp = np.pad(inp, ((0,0), (0,bbsize-np.shape(inp)[1]%bbsize),
                               (0,0)), 'edge')
            
    if np.shape(inp)[2]%bbsize==0:
        amtbbsx = np.shape(inp)[2]//bbsize
    else:
        #xmax = xmax + (bbsize - (xmax % bbsize)) #same as above.
        amtbbsx = np.shape(inp)[2]//bbsize +1
        #add zero padding
        if pad==0:
            inp = np.pad(inp, ((0,0), (0,0),
                               (0,bbsize-np.shape(inp)[2]%bbsize)), 'constant')
        elif pad==1:
            inp = np.pad(inp, ((0,0), (0,0),
                               (0,bbsize-np.shape(inp)[2]%bbsize)), 'reflect')
        else:
            inp = np.pad(inp, ((0,0), (0,0),
                               (0,bbsize-np.shape(inp)[2]%bbsize)), 'edge')

    #create a 2D array that contains the center points of all potential
    #boundary boxes. Each line represents a boundary box.
    #The first three entries are the z y x coordinates
    #(, then a boolean value. This boolean decides wether a boundary box 
    #contains values or not.)<-- deprecated, not in use any longer
    bbcntrs = np.zeros([amtbbsx*amtbbsy*amtbbsz,3],int)
    
    counter = 0
    #now fill the 3D array with the center coordinates
    for i in range(bbsize//2 -1,np.shape(inp)[0],bbsize):
        for ii in range(bbsize//2 -1,np.shape(inp)[1],bbsize):
            for iii in range(bbsize//2 -1,np.shape(inp)[2],bbsize):
                bbcntrs[counter,:] = [i, ii, iii]
                counter = counter +1
        
    #now, for each individual potential box, we want to find out if and which
    #points are contained in it
    for i in range(0,np.shape(bbcntrs)[0]):
        #find min,max values of for the current bb
        zmaxbb = bbcntrs[i,0] + bbsize//2
        zminbb = bbcntrs[i,0] - bbsize//2 +1
        ymaxbb = bbcntrs[i,1] + bbsize//2
        yminbb = bbcntrs[i,1] - bbsize//2 +1
        xmaxbb = bbcntrs[i,2] + bbsize//2
        xminbb = bbcntrs[i,2] - bbsize//2 +1
        
        #now, find all points that are inside the box
        
        #compared to the "normal" boundarysplitsimple method, we know where
        #our points are and simply have to copy the part of input matrix
        #into our bbvolume
        bbvol = inp[zminbb:zmaxbb+1, yminbb:ymaxbb+1,xminbb:xmaxbb+1]
        
        #only append a boundary box to out, if there are ones in the bb
        if np.max(bbvol>0)==1 : 
            out.append([bbvol,bbcntrs[i]])
            
    if debug==1:
        print("Amount of bbs in output " + str(len(out)) +"\n"
              + "Done.\n"
              + "Current Runtime: " + str(time.time() - starttime))
        
    return out

def computebbsize(kr):
    nthpower = int(ceil(log(2.28*kr) / log(2)))
    bbsize = 2**nthpower
    
    return bbsize

def bbs2vol(out, kr, fst=-1, lst=-1,bordervis=0,bordervar=2):
    '''
    The input is the matrix 'out' from a method that constructed boundary 
    boxes from an input shape and the kernel radius that was chosen in said 
    method. For example, simplebs_pointcloud().
    Unifies the bbs to one volume and outputs this matrix.
    
    In some instances (e.g. cell dataset and the volume would be too big) it
    is useful to limit oneself to a smaller amount of boundary boxes. This is
    what the parameters fst and lst are for. They specify the first and last
    boundary box that is chosen for the volume reconstruction.
    
    The parameter 'bordervis' is responsible for visualising the borders of
    the boundary boxes. The border values are set to 'borderval'.
    
    The output is a volume of said shape, where the boundary boxes with their
    respective 1 entries were inserted. It's called 'vol'.
    If it all works well, one should be able to see the outline of the shape
    and the border of the respective boundary boxes.
    '''
    if bordervis==1:
        dims = np.shape(out[0][0])
        for i in range(len(out)):
            out[i][0][0:dims[0],0,0]=bordervar
            out[i][0][0:dims[0],0,dims[2]-1]=bordervar
            out[i][0][0:dims[0],dims[1]-1,0]=bordervar
            out[i][0][0:dims[0],dims[1]-1,dims[2]-1]=bordervar
            out[i][0][0,0:dims[1],0]=bordervar
            out[i][0][0,0:dims[1],dims[2]-1]=bordervar
            out[i][0][dims[0]-1,0:dims[1],0]=bordervar
            out[i][0][dims[0]-1,0:dims[1],dims[2]-1]=bordervar
            out[i][0][0,0,0:dims[2]]=bordervar
            out[i][0][0,dims[1]-1,0:dims[2]]=bordervar
            out[i][0][dims[0]-1,0,0:dims[2]]=bordervar
            out[i][0][dims[0]-1,dims[1]-1,0:dims[2]]=bordervar
    
    #split input 'out' into boundary boxes 'bbs' and bb centers 'bbcntrs'
    bbs = [item[0] for item in out]
    bbs = np.asarray(bbs)
    bbcntrslist = [item[1] for item in out]
    
    nthpower = int(ceil(log(2.28*kr) / log(2)))
    bbsize = 2**nthpower
    
    #put bbcenters in an array to simplify the other steps
    bbcntrs = np.zeros([len(bbcntrslist),3],int)
    
    for i in range(0, len(bbcntrslist)):
        bbcntrs[i] = bbcntrslist[i]
        
    if fst!=-1 & lst!=-1:
        bbs = bbs[fst:lst]
        bbcntrs = bbcntrs [fst:lst]
        
    #find max,min vals, then adjust these to the bbsize to get volume dims
    # zmax = np.max(bbcntrs[:,0]) + bbsize//2
    zmin = np.min(bbcntrs[:,0]) - bbsize//2 +1
    # ymax = np.max(bbcntrs[:,1]) + bbsize//2
    ymin = np.min(bbcntrs[:,1]) - bbsize//2 +1
    # xmax = np.max(bbcntrs[:,2]) + bbsize//2
    xmin = np.min(bbcntrs[:,2]) - bbsize//2 +1
    
    #shift the coordinates accordingly
    bbcntrs = bbcntrs - [zmin,ymin,xmin]
    
    #find min max again (this could be left out, if you think more about 
    #next steps, but I can't concentrate too well rn. I might change this part
    #eventually.)
    zmax = np.max(bbcntrs[:,0])
    zmin = np.min(bbcntrs[:,0])
    ymax = np.max(bbcntrs[:,1])
    ymin = np.min(bbcntrs[:,1])
    xmax = np.max(bbcntrs[:,2])
    xmin = np.min(bbcntrs[:,2])
    
    #initalize "big" volume
    vol = np.zeros((zmax+ bbsize//2+1, ymax+ bbsize//2+1, xmax+ bbsize//2+1),
                   np.double)
    
    for i in range(0, np.shape(bbcntrs)[0]):
        idymin = bbcntrs[i,1] -bbsize//2 +1
        idymax = bbcntrs[i,1] +bbsize//2 +1
        idxmin = bbcntrs[i,2] -bbsize//2 +1
        idxmax = bbcntrs[i,2] +bbsize//2 +1
        
        for j in range (0,bbsize):
            idz = bbcntrs[i,0] -bbsize//2 +1 +j
            #idzmax = bbcntrs[i,0] +bbsize//2-1
            
            vol[idz, idymin:idymax, idxmin:idxmax] = bbs[i,j]
            
    return vol






#check the methods below, especially the indices in item
















def specialbbstovol(out, kr, fst=-1, lst=-1):
    '''
    This is boundaryboxtovol but for the mean and gauss curvature bbs method.
    
    The input is the matrix 'out' from the previously run method that
    constructed boundary boxes from an input shape and the kernel radius that
    was chosen in said method.
    
    In some instances (e.g. cell dataset and the volume would be too big) it
    is useful to limit oneself to a smaller amount of boundary boxes. This is
    what the parameters fst and lst are for. They specify the first and last
    boundary box that is chosen for the volume reconstruction.
    
    The output is a volume of said shape, where the boundary boxes with their
    respective 1 entries were inserted. It's called 'vol'.
    If it all works well, one should be able to see the outline of the shape
    and the border of the respective boundary boxes.
    '''
    #split input 'out' into boundary boxes 'bbs' and bb centers 'bbcntrs'
    bbs = [item[0] for item in out]
    bbs = np.asarray(bbs)
    bbcntrslist = [item[2] for item in out]
    
    nthpower = int(ceil(log(2.28*kr) / log(2)))
    bbsize = 2**nthpower
    
    #initialize the index of the center point of the bb
    #bbcntr = bbsize//2 -1
    
    #put bbcenters in an array to simplify the other steps
    bbcntrs = np.zeros([len(bbcntrslist),3],int)
    
    for i in range(0, len(bbcntrslist)):
        bbcntrs[i] = bbcntrslist[i]
        
    if fst!=-1 & lst!=-1:
        bbs = bbs[fst:lst]
        bbcntrs = bbcntrs [fst:lst]
        
    #find max,min vals, then adjust these to the bbsize to get volume dims
    # zmax = np.max(bbcntrs[:,0]) + bbsize//2
    zmin = np.min(bbcntrs[:,0]) - bbsize//2 +1
    # ymax = np.max(bbcntrs[:,1]) + bbsize//2
    ymin = np.min(bbcntrs[:,1]) - bbsize//2 +1
    # xmax = np.max(bbcntrs[:,2]) + bbsize//2
    xmin = np.min(bbcntrs[:,2]) - bbsize//2 +1
    
    #shift the coordinates accordingly
    bbcntrs = bbcntrs - [zmin,ymin,xmin]
    
    #find min max again (this could be left out, if you think more about 
    #next steps, but I can't concentrate too well rn. I might change this part
    #eventually.)
    zmax = np.max(bbcntrs[:,0])
    zmin = np.min(bbcntrs[:,0])
    ymax = np.max(bbcntrs[:,1])
    ymin = np.min(bbcntrs[:,1])
    xmax = np.max(bbcntrs[:,2])
    xmin = np.min(bbcntrs[:,2])
    
    #initalize "big" volume
    vol = np.zeros((zmax+ bbsize//2+1, ymax+ bbsize//2+1, xmax+ bbsize//2+1),
                   np.double)
    
    for i in range(0, np.shape(bbcntrs)[0]):
        idymin = bbcntrs[i,1] -bbsize//2 +1
        idymax = bbcntrs[i,1] +bbsize//2 +1
        idxmin = bbcntrs[i,2] -bbsize//2 +1
        idxmax = bbcntrs[i,2] +bbsize//2 +1
        
        for j in range (0,bbsize):
            idz = bbcntrs[i,0] -bbsize//2 +1 +j
            #idzmax = bbcntrs[i,0] +bbsize//2-1
            
            # print(idymin)
            # print(idymax)
            # print(idxmin)
            # print(idxmax)
            vol[idz, idymin:idymax, idxmin:idxmax] = bbs[i,j]
            
    return vol

def evcbbstovol(out, kr, idx=1, idcntr=2, fst=-1, lst=-1):
    '''
    This is boundaryboxtovol but for the surface normals of the mc and gc
    bbs method.
    
    If input is a result of pca mc or pca gc then the default index idx=1
    yields the surface normal and idcntr is bbcntr.
    
    
    The input is the matrix 'out' from the previously run method that
    constructed boundary boxes from an input shape and the kernel radius that
    was chosen in said method.
    
    In some instances (e.g. cell dataset and the volume would be too big) it
    is useful to limit oneself to a smaller amount of boundary boxes. This is
    what the parameters fst and lst are for. They specify the first and last
    boundary box that is chosen for the volume reconstruction.
    
    The output is a volume of said shape, where the boundary boxes with their
    respective 1 entries were inserted. It's called 'vol'.
    If it all works well, one should be able to see the outline of the shape
    and the border of the respective boundary boxes.
    '''
    #split input 'out' into boundary boxes 'bbs' and bb centers 'bbcntrs'
    bbs = [item[idx] for item in out] #get eigenvectors
    bbs = np.asarray(bbs)
    bbcntrslist = [item[idcntr] for item in out]
    
    nthpower = int(ceil(log(2.28*kr) / log(2)))
    bbsize = 2**nthpower
    
    #initialize the index of the center point of the bb
    #bbcntr = bbsize//2 -1
    
    #put bbcenters in an array to simplify the other steps
    bbcntrs = np.zeros([len(bbcntrslist),3],int)
    
    for i in range(0, len(bbcntrslist)):
        bbcntrs[i] = bbcntrslist[i]
        
    if fst!=-1 & lst!=-1:
        bbs = bbs[fst:lst]
        bbcntrs = bbcntrs [fst:lst]
        
    #find max,min vals, then adjust these to the bbsize to get volume dims
    # zmax = np.max(bbcntrs[:,0]) + bbsize//2
    zmin = np.min(bbcntrs[:,0]) - bbsize//2 +1
    # ymax = np.max(bbcntrs[:,1]) + bbsize//2
    ymin = np.min(bbcntrs[:,1]) - bbsize//2 +1
    # xmax = np.max(bbcntrs[:,2]) + bbsize//2
    xmin = np.min(bbcntrs[:,2]) - bbsize//2 +1
    
    #shift the coordinates accordingly
    bbcntrs = bbcntrs - [zmin,ymin,xmin]
    
    #find min max again (this could be left out, if you think more about 
    #next steps, but I can't concentrate too well rn. I might change this part
    #eventually.)
    zmax = np.max(bbcntrs[:,0])
    zmin = np.min(bbcntrs[:,0])
    ymax = np.max(bbcntrs[:,1])
    ymin = np.min(bbcntrs[:,1])
    xmax = np.max(bbcntrs[:,2])
    xmin = np.min(bbcntrs[:,2])
    
    #initalize "big" volume
    vol = np.zeros((zmax+ bbsize//2+1, ymax+ bbsize//2+1, xmax+ bbsize//2+1,3),
                   np.double)
    
    for i in range(0, np.shape(bbcntrs)[0]):
        idymin = bbcntrs[i,1] -bbsize//2 +1
        idymax = bbcntrs[i,1] +bbsize//2 +1
        idxmin = bbcntrs[i,2] -bbsize//2 +1
        idxmax = bbcntrs[i,2] +bbsize//2 +1
        
        for j in range (0,bbsize):
            idz = bbcntrs[i,0] -bbsize//2 +1 +j
            #idzmax = bbcntrs[i,0] +bbsize//2-1
            
            # print(idymin)
            # print(idymax)
            # print(idxmin)
            # print(idxmax)
            vol[idz, idymin:idymax, idxmin:idxmax,:] = bbs[i,j,:]
            
    return vol











#end