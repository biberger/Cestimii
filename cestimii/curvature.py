# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:54:39 2021

@author: Simon Biberger

Cestimii.
Compute curvature estimations on point clouds using integral invariants.

This file contains all methods that have something to do with the calculation 
of curvature estimations, such as principal curvatures, mean curvature,
gaussian curvature, etc. .
"""

import time
import numpy as np
import scipy.ndimage as nd
import scipy.signal as sg

import cestimii.occupancygrids as og
import cestimii.split as split

def cepca_discreteconvolution(OGD, OGB, xcoords, ycoords, zcoords,
                              cm, debug=0):
    """
    To calculate PCA in ball neighborhoods, we have to calculate 3 integrals:
        1) Vb = 1D \ast 1B, where 1D=OGD, 1B=OGB,
            
        2) sb = 1/Vb * (sb1, sb2, sb3)^T = 1/Vb * (1D \ast (-X*1B)
                                            
                       [1D \ast -x*1B]
              = 1/Vb * [1D \ast -y*1B]
                       [1D \ast -z*1B] , 
           where X=(x,y,z)^T, ogbx=-x*1B, ..., sbhelper1= 1D \ast -x*1B,
           sbhelper2=...
                                           
        3) Jb = (1D \ast (1B*X*X^T))- Vb*sb*sb^T
                                                   
                [1D \ast x^2*1B, 1D \ast xy*1B,   1D \ast xz*1B]
              = [1D \ast xy*1B,  1D \ast y^2*1B,  1D \ast yz*1B]
                [1D \ast xz*1B,  1D \ast yz*1B,  1D \ast z^2*1B]
                
                              [sb1^2,   sb1*sb2, sb1*sb3]
                - Vb * 1/Vb^2 [sb1*sb2, sb2^2,   sb1*sb3]
                              [sb1*sb3, sb1*sb3, sb3^2  ],
            where Jbhxx=1D \ast x^2*1B, Jbhxy=1D \ast xy*1B, ..., ogbxx=x^2*1B,
            ogbxy=xy*1B, ...
    """
    if debug==1:
        starttime = time.time() 
        
    if cm<1:
        if cm==0:
            nofftmode = 'constant' #default mode, default padding is cval=0
        elif cm==0.25:
            nofftmode = 'reflect' #reflect the edge of the last pixel
        elif cm==0.5:
            nofftmode = 'nearest' #replicate last pixel
        elif cm==0.75:
            nofftmode = 'mirror' #reflect about the center of the last pixel
        elif cm==0.95:
            nofftmode = 'wrap' #extend by wrapping around to the opposite edge
        '''
        Results of nofftmode:
            -constant: leads to heavy artifacts between the boxes. On
                       Ellipsoid these boundary artifacts also completely
                       ruin the coloring.
            -reflect: removed the above artifacts between boxes,
                      but does seem to cause artifacts on circular
                      structures, such as the "endcaps"/circles on the
                      closed cylinder dataset.
                      In a Sphere, this leads to heavy artifacts between
                      boxes. In Ellipsoid ruins the symmetry of coloring.
            -nearest: looks like reflect, but in a sphere the artifacts
                      are not quite as bad between boxes as with reflect.
                      Leads to loosing symmetry of coloring on Ellipsoid.
                      Boundary artifacts can be seen but not suuuper bad.
            -mirror: same as reflect, maybe even worse. Pretty much
                     unusable results on Ellipsoid. Ruins symmetry and
                     coloring through heavy boundary box artifacts.
            -wrap: looks like the worst of both worlds, i.e. we get
                   boundary artifacts (but on mostly on a bigger scale, as
                   if we had bigger bbs than we actually have) and the
                   artifacts on circular structures. Combined, these two
                   lead to ugly artifacts in areas that were smooth
                   previously, such as the "long sides" on the cylinder.
                   On the Sphere, it leads to unsymmetric artifacts, some
                   worse than others. In some areas it has better results
                   than above, in some worse.
                   Unusable results with ellipsoid.
                   OVERALL: NO RECOMMENDATION IN ANY CIRCUMSTANCE
        '''
        #1)
        Vb = nd.filters.convolve(OGD, OGB, mode=nofftmode)
        
        if debug==1:
            print("Calculated Vb.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        #2)
        #get the indices where Vb is 0 and nonzero
        rmv=np.where(Vb==0)
        keep=np.where(Vb!=0)
    
        sbhelperx = nd.filters.convolve(OGD, -xcoords*OGB,
                                        mode=nofftmode)
        sbhelpery = nd.filters.convolve(OGD, -ycoords*OGB,
                                        mode=nofftmode)
        sbhelperz = nd.filters.convolve(OGD, -zcoords*OGB,
                                        mode=nofftmode)
        
        if debug==1:
            print("Calculated sbhelperx,y,z.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        #3)
        sbsbt = np.array([[sbhelperx**2, sbhelperx*sbhelpery,
                           sbhelperx*sbhelperz],
                         [sbhelperx*sbhelpery, sbhelpery**2,
                          sbhelpery*sbhelperz],
                         [sbhelperx*sbhelperz, sbhelpery*sbhelperz,
                          sbhelperz**2]])
        Jbhxx = nd.filters.convolve(OGD, xcoords*xcoords*OGB,
                                        mode=nofftmode)
        Jbhyy = nd.filters.convolve(OGD, ycoords*ycoords*OGB,
                                        mode=nofftmode)
        Jbhzz = nd.filters.convolve(OGD, zcoords*zcoords*OGB,
                                        mode=nofftmode)
        Jbhxy = nd.filters.convolve(OGD, xcoords*ycoords*OGB,
                                        mode=nofftmode)
        Jbhxz = nd.filters.convolve(OGD, xcoords*zcoords*OGB,
                                        mode=nofftmode)
        Jbhyz = nd.filters.convolve(OGD, ycoords*zcoords*OGB,
                                        mode=nofftmode)
        Jb = np.array([[Jbhxx, Jbhxy, Jbhxz],
                       [Jbhxy, Jbhyy, Jbhyz],
                       [Jbhxz, Jbhyz, Jbhzz]])
        if debug==1:
            print("Calculated Jb.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return Jb, sbsbt, Vb, rmv, keep
    else:
        raise NameError('Wrong cm parameter in cepca_discreteconvolution.')
        
def cepca_fft(OGD, OGB, xcoords, ycoords, zcoords, cm, debug=0):
    """
    To calculate PCA in ball neighborhoods, we have to calculate 3 integrals:
        1) Vb = 1D \ast 1B, where 1D=OGD, 1B=OGB,
            
        2) sb = 1/Vb * (sb1, sb2, sb3)^T = 1/Vb * (1D \ast (-X*1B)
                                            
                       [1D \ast -x*1B]
              = 1/Vb * [1D \ast -y*1B]
                       [1D \ast -z*1B] , 
           where X=(x,y,z)^T, ogbx=-x*1B, ..., sbhelper1= 1D \ast -x*1B,
           sbhelper2=...
                                           
        3) Jb = (1D \ast (1B*X*X^T))- Vb*sb*sb^T
                                                   
                [1D \ast x^2*1B, 1D \ast xy*1B,   1D \ast xz*1B]
              = [1D \ast xy*1B,  1D \ast y^2*1B,  1D \ast yz*1B]
                [1D \ast xz*1B,  1D \ast yz*1B,  1D \ast z^2*1B]
                
                              [sb1^2,   sb1*sb2, sb1*sb3]
                - Vb * 1/Vb^2 [sb1*sb2, sb2^2,   sb1*sb3]
                              [sb1*sb3, sb1*sb3, sb3^2  ],
            where Jbhxx=1D \ast x^2*1B, Jbhxy=1D \ast xy*1B, ..., ogbxx=x^2*1B,
            ogbxy=xy*1B, ...
    """
    if debug==1:
        starttime = time.time() 
        
    if cm>=1:
        #1)
        Vb = sg.fftconvolve(OGD, OGB, mode='same')
        
        if debug==1:
            print("Calculated Vb with FFT.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        #2)
        #get the indices where Vb is 0 and nonzero
        rmv=np.where(Vb==0)
        keep=np.where(Vb!=0)
        
        sbhelperx = sg.fftconvolve(OGD, -xcoords*OGB, mode='same')
        sbhelpery = sg.fftconvolve(OGD, -ycoords*OGB, mode='same')
        sbhelperz = sg.fftconvolve(OGD, -zcoords*OGB, mode='same')
        
        if debug==1:
            print("Calculated sbhelperx,y,z with FFT.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        #3)
        sbsbt = np.array([[sbhelperx**2, sbhelperx*sbhelpery,
                           sbhelperx*sbhelperz],
                         [sbhelperx*sbhelpery, sbhelpery**2,
                          sbhelpery*sbhelperz],
                         [sbhelperx*sbhelperz, sbhelpery*sbhelperz,
                          sbhelperz**2]])
        Jbhxx = sg.fftconvolve(OGD, xcoords*xcoords*OGB, mode='same')
        Jbhyy = sg.fftconvolve(OGD, ycoords*ycoords*OGB, mode='same')
        Jbhzz = sg.fftconvolve(OGD, zcoords*zcoords*OGB, mode='same')
        Jbhxy = sg.fftconvolve(OGD, xcoords*ycoords*OGB, mode='same')
        Jbhxz = sg.fftconvolve(OGD, xcoords*zcoords*OGB, mode='same')
        Jbhyz = sg.fftconvolve(OGD, ycoords*zcoords*OGB, mode='same')
        
        Jb = np.array([[Jbhxx, Jbhxy, Jbhxz],
                       [Jbhxy, Jbhyy, Jbhyz],
                       [Jbhxz, Jbhyz, Jbhzz]])
        if debug==1:
            print("Calculated Jb.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
        
        return Jb, sbsbt, Vb, rmv, keep
    else:
        raise NameError('Wrong cm parameter in cepca_discreteconvolution.')
 
def cepca_orderevals_size(jbev1, jbev2, jbev3, jbevc1, jbevc2, jbevc3):
    '''
    In the new version we take the eigenvalues that are given bei "eigh" in
    ascending order and order them in descending order.
    
    In the outdated version we forcefully check the order:
    We need to know where the following cases appear. 
    Where is jbev1>=jbev2, jbev1>=jbev3, jbev2>=jbev3?
    (in the cases below, don't substitute the where with
    np.where(), but with the matrix inequality operator)
    -1) If all these are true, then we don't have to change the 
    order. 
    (111) -> np.where[1*1*1=1 >0]
    -2) If the first two are true and the last one is false, then
    change the order of k2 and k3. 
    (110) -> np.where[multiply ( (1*1=1) AND (where x<1) )>0]
    *-3) If the first and the last one are true, second one false,
    then there is an error as this violates transitivity of the
    >= operator. There's no need to test for this case. (101)
    -4) If the first is true, second and third ones are false,
    then change the order so k3 is first, k1 is second and k2
    is third. 
    (100) -> np.where[multiply ( (where x>0) AND (where (0*0=0)<1) )>0]
    -5) If the first is false, second and third ones are true,
    then change the order of k1 and k2.
    (011) -> np.where[multiply ( (where x<1) AND (1*1=1) )>0]
    *-6) If the first and the last one are false, second one true,
    then there is an error as this violates transitivity of the
    >= operator. There's no need to test for this case. (010)
    -7) If the first two are false and the last one is true, then
    change the order so k2 is first, k3 is second and k1
    is third. 
    (001) -> np.where[multiply ( (where (0*0=0)<1) AND (where x>0) )>0]
    -8) If all these are false, then reverse the order. 
    (000) -> np.where[0*0*0=0<1]
    '''
    #new "smart" version
    
    #reorder evals, first eigenvalue to third, and third to first.
    temp = jbev1
    jbev1 = jbev3
    jbev3 = temp
    
    #reorder eigenvectors, first to third, and third to first.
    temp = jbevc1
    jbevc1 = jbevc3
    jbevc3 = temp
    
    #old version that checks everything if in correct order
    
    # #initialize test matrices (ineq1 for inequality 1, etc.)
    # ineq1 = (jbev1 - jbev2) >=0
    # ineq2 = (jbev1 - jbev3) >=0
    # ineq3 = (jbev2 - jbev3) >=0
    
    # #multiplications that are needed several times (this is for 
    # #optimization purposes)
    # multXXY = ineq1*ineq2
    # multYXX = ineq2*ineq3
    
    # #get indices for the test cases (c1 for case 1, etc.)
    # #(111) -> np.where[1*1*1=1 >0]
    # #c1 = np.where(multXXY * ineq3 >0) 
    # #we don't need the first case as there is no change
    # #in the order
    # #(110) -> np.where[multiply ( (1*1=1) AND (where x<1) )]
    # c2 = np.where((multXXY * (ineq3<1)) >0)
    # #(100) -> np.where[multiply ( (where x>0) AND (where (0*0=0)<1) )>0]
    # c4 = np.where((ineq1 * (multYXX<1)) >0)
    # #(011) -> np.where[multiply ( (where x<1) AND (1*1=1) )>0]
    # c5 = np.where(((ineq1<1) * multYXX) >0)
    # #(001) -> np.where[multiply ( (where (0*0=0)<1) AND (where x>0) )>0]
    # c7 = np.where(((multXXY<1) * ineq3) >0)
    # #(000) -> np.where[0*0*0=0<1]
    # c8 = np.where((ineq1 * multYXX) <1)
    
    # #handle the cases
    # temp = np.zeros(np.shape(jbev1)) #temporary value saver
    # tempvec = np.zeros(np.shape(jbevc1)) #temporary value saver for eigvecs
    
    # #case 2, change order of k2, k3
    # temp[c2] = jbev2[c2]
    # jbev2[c2] = jbev3[c2]
    # jbev3[c2] = temp[c2]
    # #and now the eigenvalues
    # tempvec[c2] = jbevc2[c2]
    # jbevc2[c2] = jbevc3[c2]
    # jbevc3[c2] = tempvec[c2]
    
    # #case 4, change order to k3, k1, k2
    # #switch k2,k3 first, then k1, k3
    # temp[c4] = jbev2[c4]
    # jbev2[c4] = jbev3[c4]
    # jbev3[c4] = temp[c4]
    # #and now the eigenvalues
    # tempvec[c4] = jbevc2[c4]
    # jbevc2[c4] = jbevc3[c4]
    # jbevc3[c4] = tempvec[c4]
    # #k1 is still in k1 position, k3 is now in k2 position
    # temp[c4] = jbev1[c4]
    # jbev1[c4] = jbev2[c4]
    # jbev2[c4] = temp[c4]
    # #and now the eigenvalues
    # tempvec[c4] = jbevc1[c4]
    # jbevc1[c4] = jbevc2[c4]
    # jbevc2[c4] = tempvec[c4]
    
    # #case 5, change order of k1, k2
    # temp[c5] = jbev1[c5]
    # jbev1[c5] = jbev2[c5]
    # jbev2[c5] = temp[c5]
    # #and now the eigenvalues
    # tempvec[c5] = jbevc1[c5]
    # jbevc1[c5] = jbevc2[c5]
    # jbevc2[c5] = tempvec[c5]
    
    # #case 7, change order to k2, k3, k1
    # #switch k1,k2 first, then k1, k3
    # temp[c7] = jbev1[c7]
    # jbev1[c7] = jbev2[c7]
    # jbev2[c7] = temp[c7]
    # #and now the eigenvalues
    # tempvec[c7] = jbevc1[c7]
    # jbevc1[c7] = jbevc2[c7]
    # jbevc2[c7] = tempvec[c7]
    # #k3 is still in k3 position, k1 is now in k2 position
    # temp[c7] = jbev3[c7]
    # jbev3[c7] = jbev2[c7]
    # jbev2[c7] = temp[c7]
    # #and now the eigenvalues
    # tempvec[c7] = jbevc3[c7]
    # jbevc3[c7] = jbevc2[c7]
    # jbevc2[c7] = tempvec[c7]
    
    # #case 8, reverse order, i.e. k3, k2, k1. Swap k3 with k1.
    # temp[c8] = jbev1[c8]
    # jbev1[c8] = jbev3[c8]
    # jbev3[c8] = temp[c8]  
    # #and now the eigenvalues
    # tempvec[c8] = jbevc1[c8]
    # jbevc1[c8] = jbevc3[c8]
    # jbevc3[c8] = tempvec[c8]
    
    return jbev1, jbev2, jbev3, jbevc1, jbevc2, jbevc3

def cepca_orderevals_error(kr, jbev1, jbev2, jbev3, jbevc1, jbevc2, jbevc3):
    '''
    Our problem is that the above eigenvalue matrix is not ordered
    in a meaningful way, whereas the eigenvalues in the Pottmann07
    paper are ordered, but in a unknown way.
    One can assume, that the order is given by the biggest
    elements being first (this assumption is handled in the above
    if case). If one does not assume this, then there's not much
    left to do except using the error and value estimates given
    in the paper and looking for the right combination of 
    eigenvalues that fulfills these approximation.
    One can then apply a method of choosing the combination with
    the lowest amount of errors, for example by just summing up
    all errors, averaging them and choosing the combination with
    lowest average.
    This is what is done here.
    We have the three eigenvalues kappa1, kappa2, kappa3 from
    above and either kappa1,kappa2,   kappa1,kappa3,   or 
    kappa2,kappa3 are the two eigenvalues referenced in the paper
    to calculate the principal curvatures Kappa1, Kappa2. Note,
    that in the above combinations the order can be switched, but
    this does not lead to different results in the terms below 
    (except in 4.1), 4.2), but we'll deal with that by 
    calculating the error and averaging smart).
    
    We know,
    1) Kappa1 = 6/(pi*kr**6) * (kappa2-3*kappa1) + 8/(5*kr) ,
    2) Kappa2 = 6/(pi*kr**6) * (kappa1-3*kappa2) + 8/(5*kr) ,
    3) kappa2-kappa1 = pi/24 * (Kappa1-Kappa2) * kr**6 + O(kr**7),
    4.1) kappa1 = (2*pi)/15 * kr**5 
              - (pi/48) * (3*Kappa1+Kappa2) * kr**6 + O(kr**7)
    4.2) kappa2 = (2*pi)/15 * kr**5 
              - (pi/48) * (Kappa1+3*Kappa2) * kr**6 + O(kr**7)
    4.3) kappa3 = (19*pi)/480 * kr**5 
              - (9*pi/512) * (Kappa1+Kappa2) * kr**6 + O(kr**7)
    
    So, to get the error estimates, we will calculate Kappa1 and
    Kappa2, i.e. 1)&2), for all three combinations and then compare
    the difference of the two terms in 3), and then compare kappa1-3
    to their respective terms in 4.1)-4.3).
    Then, for each combination we sum up the above differences,
    take the average.
    
    Because the eigenvalues are saved in matrices and these might
    be big, we won't create several matrices, but keep the ones
    we have right now and change the order in the current one.
    Therefore we only need one matrix to temporarily save values.
    '''
    error=np.zeros(np.shape(jbev1)+(3,), np.double) #contains 
                                                 #error estimates
    temp = np.zeros(np.shape(jbev1)) #temporary value saver for eigvals
    tempvec = np.zeros(np.shape(jbevc1)) #temporary value saver for eigvecs
    
    for i in range(0,3):
        #calculate the errors for the current order, i.e. k1,k2
        Kappa1 = 6/(np.pi*kr**6) * (jbev2-3*jbev1) + 8/(5*kr) #1)
        Kappa2 = 6/(np.pi*kr**6) * (jbev1-3*jbev2) + 8/(5*kr) #2)
        
        err1lft = jbev2-jbev1 #3) left side
        err1rgh = np.pi/24 * (Kappa1-Kappa2) * kr**6 #3) right side
        
        #weird formatting due to some strange spyder behavior,
        #this resolves after a restart hopefully. Then I'll change
        #the formatting here.
        err2kap1 = (2*np.pi)/15 * kr**5\
                    - (np.pi/48) * (3*Kappa1+Kappa2) * kr**6 
                    #4.1) right side
        err2kap2 = (2*np.pi)/15 * kr**5\
                    - (np.pi/48) * (Kappa1+3*Kappa2) * kr**6 
                    #4.2) right side
        err2kap3 = (19*np.pi)/480 * kr**5 \
                    - (9*np.pi/512) * (Kappa1+Kappa2) * kr**6 
                    #4.3) right side
                  
        error[:,:,:,i] = 0.25*np.abs(err1lft-err1rgh)\
                         + 0.5*np.abs((jbev1+jbev2)\
                         -(err2kap1+err2kap2))\
                         + 0.25*np.abs(jbev3-err2kap3)
                 
        #now to the swapping of eigenvalues, i.e. try a different
        #combination
        if i==0:
            #now we swap jbev 2 with jbev3.
            temp = jbev2
            jbev2 = jbev3
            jbev3 = temp
        elif i==1:
            #now we swap jbev 1 with what used to be jbev2 (now
            #jbev3, position-wise)
            temp = jbev1
            jbev1 = jbev3
            jbev3 = temp
        
    #now we compare errors!
    #first case is where the initial order k1,k2,k3 has the lowest
    #error
    case1 = np.where((error[:,:,:,0]-error[:,:,:,1]<=0) 
                 * (error[:,:,:,0]-error[:,:,:,2]<=0))
    #2nd case is where the order k1,k3,k2 has the lowest error
    case2 = np.where((error[:,:,:,1]-error[:,:,:,0] <=0)
                 * (error[:,:,:,1]-error[:,:,:,2]<=0))
    #3rd case is where the order k2,k3,k1 has the lowest error
    case3 = np.where((error[:,:,:,2]-error[:,:,:,0]<=0)
                 * (error[:,:,:,2]-error[:,:,:,1] <=0))
    #we need case 3 for the eigenvectors
    
    #eigenvalue switching
    #wherever case1,case2 is true, we have to switch the order
    #back. In case 3, nothing has to happen.
    #for case 2
    temp[case2] = jbev1[case2] #current order is k2,k3,k1
    jbev1[case2] = jbev3[case2] #k1,k3,k1
    jbev3[case2] = temp[case2] #k1,k2,k2
    
    #for case 1
    temp[case1] = jbev1[case1] #current order is k2,k3,k1
    jbev1[case1] = jbev3[case1] #k1,k3,k1
    jbev3[case1] = temp[case1] #k1,k3,k2
    temp[case1] = jbev2[case1]
    jbev2[case1] = jbev3[case1] #k1,k2,k2
    jbev3[case1] = temp[case1] #k1,k2,k3
    
    #eigenvector switching
    #wherever case2,case3 is true, we have to switch the order
    #As the eigenvectors weren't rearranged yet, in case 1 nothing
    #happens.
    #for case 2 (we want k1,k3,k2)
    tempvec[case2] = jbevc2[case2] #current order is k1,k2,k3
    jbevc2[case2] = jbevc3[case2] #k1,k3,k3
    jbevc3[case2] = tempvec[case2] #k1,k3,k2
    
    #for case 3 (we want k2,k3,k1)
    tempvec[case3] = jbevc2[case3] #current order is k1,k2,k3
    jbevc2[case3] = jbevc3[case3] #k1,k3,k3
    jbevc3[case3] = tempvec[case3] #k1,k3,k2
    tempvec[case3] = jbevc1[case3] 
    jbevc1[case3] = jbevc3[case3] #k2,k3,k2
    jbevc3[case3] = tempvec[case3] #k2,k3,k1
        
    return jbev1, jbev2, jbev3, jbevc1, jbevc2, jbevc3

def cepca_orderevals_cdiffnrmls(OGD, cm, jbev1, jbev2, jbev3, jbevc1, jbevc2,
                                jbevc3):
    '''
    Use central differences to approximate a gradient per pixel.
    Then calculate the normal of said gradient and compare that to the
    eigenvectors. Choose the third eigenvector according to the lowest
    error between the approximated normal vector and the eigenvectors.
    
    If cm==2, use the already padded version of bbs[i] called OGD,
    otherwise, do some padding using np.pad .
    '''
    stencil = 5 #use 5x5x5 stencil for central differences
    
    if cm<1:
        if cm==0:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'constant')
            #pads with constant 0
        elif cm==0.25:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)), 
                         'reflect')
            #pads with the reflection of the vector mirrored on the 
            #first and last values of the vector along each axis
        elif cm==0.5:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'edge')
            #pads with the edge values of array.
        elif cm==0.75:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'edge')
            #pads with the edge values of array.
        elif cm==0.95:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'wrap')
            #pads with the wrap of the vector along the axis.
            #The first values are used to pad the end and the end 
            #values are used to pad the beginning.
    elif cm==1:
        OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'constant')
            #pads with constant 0
    elif cm==1.5:
        OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)), 
                         'reflect')
            #pads with the reflection of the vector mirrored on the 
            #first and last values of the vector along each axis
        # OGD = np.pad(bbs[i], ((stencil//2,stencil//2),
        #                           (stencil//2,stencil//2),
        #                           (stencil//2,stencil//2)),
        #                  'edge')
        #     #pads with the edge values of array.
    
    ogd0 = np.shape(OGD)[0]
    ogd1 = np.shape(OGD)[1]
    ogd2 = np.shape(OGD)[2]
            
    #calculate the gradient through central differences
    cdiffz = OGD[stencil-1:ogd0, stencil//2:ogd1-stencil//2,
                 stencil//2:ogd2-stencil//2] \
           - OGD[0:ogd0-stencil+1, stencil//2:ogd1-stencil//2,
                 stencil//2:ogd2-stencil//2]
    cdiffy = OGD[stencil//2:ogd0-stencil//2, stencil-1:ogd1,
                 stencil//2:ogd2-stencil//2] \
           - OGD[stencil//2:ogd0-stencil//2, 0:ogd1-stencil+1,
                 stencil//2:ogd2-stencil//2]
    cdiffx = OGD[stencil//2:ogd0-stencil//2, 
                 stencil//2:ogd1-stencil//2,
                 stencil-1:ogd2] \
           - OGD[stencil//2:ogd0-stencil//2,
                 stencil//2:ogd1-stencil//2,
                 0:ogd2-stencil+1]
    #normalization
    cdiffnorm = np.sqrt(cdiffx**2 + cdiffy**2 + cdiffz**2)
    zrs = np.where(cdiffnorm==0)
    cdiffnorm[zrs] = 1
    cdiffz = cdiffz/cdiffnorm
    cdiffy = cdiffy/cdiffnorm
    cdiffx = cdiffx/cdiffnorm
    cdiffz[zrs] = 0
    cdiffy[zrs] = 0
    cdiffx[zrs] = 0
           
    #infinitely many solutions for normal, we choose
    #n1=-z, n2=-z, n3=x+y. There's a note on the rmkbl for more details
    normalx = -cdiffz
    normaly = -cdiffz
    normalz = -(cdiffx + cdiffy)
    
    #normalization
    normalnorm = np.sqrt(normalx**2 + normaly**2 + normalz**2)
    zrs = np.where(normalnorm==0)
    normalnorm[zrs] = 1
    normalz = normalz/normalnorm
    normaly = normaly/normalnorm
    normalx = normalx/normalnorm
    normalz[zrs] = 0
    normaly[zrs] = 0
    normalx[zrs] = 0
    
    #normalise the eigenvectors
    evnorm = np.sqrt(jbevc1[:,:,:,0]**2 + jbevc1[:,:,:,1]**2 
                                     + jbevc1[:,:,:,2]**2)
    zrs = np.where(evnorm==0)
    evnorm[zrs] = 1
    jbevc1[:,:,:,0] = jbevc1[:,:,:,0]/evnorm
    jbevc1[:,:,:,1] = jbevc1[:,:,:,1]/evnorm
    jbevc1[:,:,:,2] = jbevc1[:,:,:,2]/evnorm
    jbevc1[zrs] = 0
    
    evnorm = np.sqrt(jbevc2[:,:,:,0]**2 + jbevc2[:,:,:,1]**2 
                                     + jbevc2[:,:,:,2]**2)
    zrs = np.where(evnorm==0)
    evnorm[zrs] = 1
    jbevc2[:,:,:,0] = jbevc2[:,:,:,0]/evnorm
    jbevc2[:,:,:,1] = jbevc2[:,:,:,1]/evnorm
    jbevc2[:,:,:,2] = jbevc2[:,:,:,2]/evnorm
    jbevc2[zrs] = 0
    
    evnorm = np.sqrt(jbevc3[:,:,:,0]**2 + jbevc3[:,:,:,1]**2 
                                     + jbevc3[:,:,:,2]**2)
    zrs = np.where(evnorm==0)
    evnorm[zrs] = 1
    jbevc3[:,:,:,0] = jbevc3[:,:,:,0]/evnorm
    jbevc3[:,:,:,1] = jbevc3[:,:,:,1]/evnorm
    jbevc3[:,:,:,2] = jbevc3[:,:,:,2]/evnorm
    jbevc3[zrs] = 0
    
    #if two vectors are parallel, inner product is 1
    #we now measure which eigenvector is the furthest away from zero or
    #which eigenvector is closest to 1. 
    #we calculate the errors similar to order==2.
    #We take the absolute value because we don't care about inner or outer
    #orientation
    
    #choose variant
    distvariant = "furthestfrom0"
    #distvariant = "closestto1"
    
    dist = np.zeros(np.shape(jbevc1))
    
    dist[:,:,:,0] = np.abs(jbevc1[:,:,:,0]*normalx + jbevc1[:,:,:,1]*normaly 
                           + jbevc1[:,:,:,2]*normalz)
    dist[:,:,:,1] = np.abs(jbevc2[:,:,:,0]*normalx + jbevc2[:,:,:,1]*normaly 
                           + jbevc2[:,:,:,2]*normalz)
    dist[:,:,:,2] = np.abs(jbevc3[:,:,:,0]*normalx + jbevc3[:,:,:,1]*normaly 
                           + jbevc3[:,:,:,2]*normalz)
    
    if distvariant == "furthestfrom0":
        #check which ev is the most unperpendicular,i.e. biggest dist to 0
        ev1best = (dist[:,:,:,0] >= dist[:,:,:,1]) * (dist[:,:,:,0] > dist[:,:,:,2])
        ev2best = (dist[:,:,:,1] > dist[:,:,:,0]) * (dist[:,:,:,1] > dist[:,:,:,2])
        #ev3biggest = (dist[2] > dist[0]) * (dist[2] > dist[1])
        #if ev3 is the biggest, don't do anything.
    elif distvariant == "closestto1":
        #check which ev is the most parallel,i.e. smallest dist to 1
        ev1best =((np.abs(1-dist[:,:,:,0])-np.abs(1-dist[:,:,:,1]))<=0)\
                   *((np.abs(1-dist[:,:,:,0])-np.abs(1-dist[:,:,:,2]))<0)
        ev2best =((np.abs(1-dist[:,:,:,1])-np.abs(1-dist[:,:,:,0]))<0)\
                   *((np.abs(1-dist[:,:,:,1])-np.abs(1-dist[:,:,:,2]))<0)
        #if ev3 is the closest to 1, don't do anything.
    
    #where the resp. ev's dists are the biggest, set to ev3
    ev3setter1 = np.where(ev1best==1)
    ev3setter2 = np.where(ev2best==1)
    
    temp = np.zeros(np.shape(jbev1))
    tempvec = np.zeros(np.shape(jbevc1))
    
    #swap jbev1,jbevc1 with jbev3,jbevc3
    temp[ev3setter1] = jbev3[ev3setter1]
    jbev3[ev3setter1] = jbev1[ev3setter1]
    jbev1[ev3setter1] = temp[ev3setter1]
    
    tempvec[ev3setter1] = jbevc3[ev3setter1]
    jbevc3[ev3setter1] = jbevc1[ev3setter1]
    jbevc1[ev3setter1] = tempvec[ev3setter1]
    
    #swap jbev2,jbevc2 with jbev3,jbevc3
    temp[ev3setter2] = jbev3[ev3setter2]
    jbev3[ev3setter2] = jbev2[ev3setter2]
    jbev2[ev3setter2] = temp[ev3setter2]
    
    tempvec[ev3setter2] = jbevc3[ev3setter2]
    jbevc3[ev3setter2] = jbevc2[ev3setter2]
    jbevc2[ev3setter2] = tempvec[ev3setter2]
    
    OGD = OGD[stencil//2:ogd0-stencil//2, stencil//2:ogd1-stencil//2,
              stencil//2:ogd2-stencil//2]
        
    return OGD, jbev1, jbev2, jbev3, jbevc1, jbevc2, jbevc3

def cepca_orderevals_errnrmlsmix(OGD, cm, kr, jbev1, jbev2, jbev3, jbevc1, 
                                 jbevc2, jbevc3):
    '''
    Mix of order==2, order==3. We do both the error approximation
    with the formulas AND the gradient approximation and weigh them
    49/51 to avoid patt situations, i.e. the gradient approx weighs
    more.
    
    The errors itself are on different scales so they are normalised
    so we can compare them.
    Then we put the errors together, find the lowest one and choose
    that as our third eigenvector.
    '''
    error=np.zeros(np.shape(jbev1)+(3,), np.double) #contains 
                                                     #error estimates
    temp = np.zeros(np.shape(jbev1)) #temporary value saver for eigvals
    tempvec = np.zeros(np.shape(jbevc1)) #temporary value saver for eigvecs
    
    for ii in range(0,3):
        #calculate the errors for the current order, i.e. k1,k2
        Kappa1 = 6/(np.pi*kr**6) * (jbev2-3*jbev1) + 8/(5*kr) #1)
        Kappa2 = 6/(np.pi*kr**6) * (jbev1-3*jbev2) + 8/(5*kr) #2)
        
        err1lft = jbev2-jbev1 #3) left side
        err1rgh = np.pi/24 * (Kappa1-Kappa2) * kr**6 #3) right side
        
        #weird formatting due to some strange spyder behavior,
        #this resolves after a restart hopefully. Then I'll change
        #the formatting here.
        err2kap1 = (2*np.pi)/15 * kr**5\
                    - (np.pi/48) * (3*Kappa1+Kappa2) * kr**6 
                    #4.1) right side
        err2kap2 = (2*np.pi)/15 * kr**5\
                    - (np.pi/48) * (Kappa1+3*Kappa2) * kr**6 
                    #4.2) right side
        err2kap3 = (19*np.pi)/480 * kr**5 \
                    - (9*np.pi/512) * (Kappa1+Kappa2) * kr**6 
                    #4.3) right side
                  
        error[:,:,:,ii] = 0.25*np.abs(err1lft-err1rgh)\
                     + 0.5*np.abs((jbev1+jbev2)\
                                  -(err2kap1+err2kap2))\
                     + 0.25*np.abs(jbev3-err2kap3)
                     
        #now to the swapping of eigenvalues, i.e. try a different
        #combination
        if ii==0:
            #now we swap jbev 2 with jbev3.
            temp = jbev2
            jbev2 = jbev3
            jbev3 = temp
        elif ii==1:
            #now we swap jbev 1 with what used to be jbev2 (now
            #jbev3, position-wise)
            temp = jbev1
            jbev1 = jbev3
            jbev3 = temp
    
    stencil = 5 #use 5x5x5 stencil for central differences
    
    if cm<1:
        if cm==0:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'constant')
            #pads with constant 0
        elif cm==0.25:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)), 
                         'reflect')
            #pads with the reflection of the vector mirrored on the 
            #first and last values of the vector along each axis
        elif cm==0.5:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'edge')
            #pads with the edge values of array.
        elif cm==0.75:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'edge')
            #pads with the edge values of array.
        elif cm==0.95:
            OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'wrap')
            #pads with the wrap of the vector along the axis.
            #The first values are used to pad the end and the end 
            #values are used to pad the beginning.
    elif cm==1:
        OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)),
                         'constant')
            #pads with constant 0
    elif cm==1.5:
        OGD = np.pad(OGD, ((stencil//2,stencil//2),
                                  (stencil//2,stencil//2),
                                  (stencil//2,stencil//2)), 
                         'reflect')
            #pads with the reflection of the vector mirrored on the 
            #first and last values of the vector along each axis
        # OGD = np.pad(bbs[i], ((stencil//2,stencil//2),
        #                           (stencil//2,stencil//2),
        #                           (stencil//2,stencil//2)),
        #                  'edge')
        #     #pads with the edge values of array.
    
    ogd0 = np.shape(OGD)[0]
    ogd1 = np.shape(OGD)[1]
    ogd2 = np.shape(OGD)[2]
            
    #calculate the gradient through central differences
    cdiffz = OGD[stencil-1:ogd0, stencil//2:ogd1-stencil//2,
                 stencil//2:ogd2-stencil//2] \
           - OGD[0:ogd0-stencil+1, stencil//2:ogd1-stencil//2,
                 stencil//2:ogd2-stencil//2]
    cdiffy = OGD[stencil//2:ogd0-stencil//2, stencil-1:ogd1,
                 stencil//2:ogd2-stencil//2] \
           - OGD[stencil//2:ogd0-stencil//2, 0:ogd1-stencil+1,
                 stencil//2:ogd2-stencil//2]
    cdiffx = OGD[stencil//2:ogd0-stencil//2, 
                 stencil//2:ogd1-stencil//2,
                 stencil-1:ogd2] \
           - OGD[stencil//2:ogd0-stencil//2,
                 stencil//2:ogd1-stencil//2,
                 0:ogd2-stencil+1]
    #normalization
    cdiffnorm = np.sqrt(cdiffx**2 + cdiffy**2 + cdiffz**2)
    zrs = np.where(cdiffnorm==0)
    cdiffnorm[zrs] = 1
    cdiffz = cdiffz/cdiffnorm
    cdiffy = cdiffy/cdiffnorm
    cdiffx = cdiffx/cdiffnorm
    cdiffz[zrs] = 0
    cdiffy[zrs] = 0
    cdiffx[zrs] = 0
           
    #two solutions for normal, n1=-z or z, n2=-z or z, n3=x+y or -(x+y) 
    normalx = -cdiffz
    normaly = -cdiffz
    normalz = -(cdiffx + cdiffy)
    
    #normalization
    normalnorm = np.sqrt(normalx**2 + normaly**2 + normalz**2)
    zrs = np.where(normalnorm==0)
    normalnorm[zrs] = 1
    normalz = normalz/normalnorm
    normaly = normaly/normalnorm
    normalx = normalx/normalnorm
    normalz[zrs] = 0
    normaly[zrs] = 0
    normalx[zrs] = 0
    
    #normalise the eigenvectors
    evnorm = np.sqrt(jbevc1[:,:,:,0]**2 + jbevc1[:,:,:,1]**2 
                                     + jbevc1[:,:,:,2]**2)
    zrs = np.where(evnorm==0)
    evnorm[zrs] = 1
    jbevc1[:,:,:,0] = jbevc1[:,:,:,0]/evnorm
    jbevc1[:,:,:,1] = jbevc1[:,:,:,1]/evnorm
    jbevc1[:,:,:,2] = jbevc1[:,:,:,2]/evnorm
    jbevc1[zrs] = 0
    
    evnorm = np.sqrt(jbevc2[:,:,:,0]**2 + jbevc2[:,:,:,1]**2 
                                     + jbevc2[:,:,:,2]**2)
    zrs = np.where(evnorm==0)
    evnorm[zrs] = 1
    jbevc2[:,:,:,0] = jbevc2[:,:,:,0]/evnorm
    jbevc2[:,:,:,1] = jbevc2[:,:,:,1]/evnorm
    jbevc2[:,:,:,2] = jbevc2[:,:,:,2]/evnorm
    jbevc2[zrs] = 0
    
    evnorm = np.sqrt(jbevc3[:,:,:,0]**2 + jbevc3[:,:,:,1]**2 
                                     + jbevc3[:,:,:,2]**2)
    zrs = np.where(evnorm==0)
    evnorm[zrs] = 1
    jbevc3[:,:,:,0] = jbevc3[:,:,:,0]/evnorm
    jbevc3[:,:,:,1] = jbevc3[:,:,:,1]/evnorm
    jbevc3[:,:,:,2] = jbevc3[:,:,:,2]/evnorm
    jbevc3[zrs] = 0
    
    #if two vectors are parallel, inner product is 1
    #we now measure which eigenvector is the furthest away from zero
    #we calculate the errors similar to order==2.
    dist = np.zeros(np.shape(jbevc1))
    
    dist[:,:,:,0] = np.abs(jbevc1[:,:,:,0]*normalx + jbevc1[:,:,:,1]*normaly 
                           + jbevc1[:,:,:,2]*normalz)
    dist[:,:,:,1] = np.abs(jbevc2[:,:,:,0]*normalx + jbevc2[:,:,:,1]*normaly 
                           + jbevc2[:,:,:,2]*normalz)
    dist[:,:,:,2] = np.abs(jbevc3[:,:,:,0]*normalx + jbevc3[:,:,:,1]*normaly 
                           + jbevc3[:,:,:,2]*normalz)
    
    #find biggest distance for normalization
    ev1biggest = (dist[:,:,:,0] >= dist[:,:,:,1]) * (dist[:,:,:,0] >= dist[:,:,:,2])
    ev2biggest = (dist[:,:,:,1] >= dist[:,:,:,0]) * (dist[:,:,:,1] >= dist[:,:,:,2])
    ev3biggest = (dist[:,:,:,2] >= dist[:,:,:,0]) * (dist[:,:,:,2] >= dist[:,:,:,1])
    
    #find max dist indices
    fstbig = np.where(ev1biggest==1)
    sndbig = np.where(ev2biggest==1) 
    thrbig = np.where(ev3biggest==1) 
            
    #normalise dist errors
    zrs = np.where(dist==0)
    dist[zrs] = 1
    dist[fstbig] = dist[fstbig] / dist[fstbig]
    dist[sndbig] = dist[sndbig] / dist[sndbig]
    dist[thrbig] = dist[thrbig] / dist[thrbig]
    dist[zrs] = 0
    
    #find biggest error for normalization
    err1biggest = (error[:,:,:,0] >= error[:,:,:,1]) * (error[:,:,:,0] >= error[:,:,:,2])
    err2biggest = (error[:,:,:,1] >= error[:,:,:,0]) * (error[:,:,:,1] >= error[:,:,:,2])
    err3biggest = (error[:,:,:,2] >= error[:,:,:,0]) * (error[:,:,:,2] >= error[:,:,:,1])
    
    #find max dist indices
    fstbig = np.where(err1biggest==1)
    sndbig = np.where(err2biggest==1) 
    thrbig = np.where(err3biggest==1) 
            
    #normalise dist errors
    zrs = np.where(error==0)
    error[zrs] = 1
    error[fstbig] = error[fstbig] / error[fstbig]
    error[sndbig] = error[sndbig] / error[sndbig]
    error[thrbig] = error[thrbig] / error[thrbig]
    error[zrs] = 0
    
    #subtract dist from error, because the higher the dist, the better
    #we get issues with the below error comparison if error has
    #negative values, but due to the normalisation of both, this
    #should not be able to happen.
    error = error-dist
    
    #now we compare errors!
    #first case is where the initial order k1,k2,k3 has the lowest
    #error
    case1 = np.where((error[:,:,:,0]-error[:,:,:,1]<=0) 
                     * (error[:,:,:,0]-error[:,:,:,2]<=0))
    #2nd case is where the order k1,k3,k2 has the lowest error
    case2 = np.where((error[:,:,:,1]-error[:,:,:,0] <=0)
                     * (error[:,:,:,1]-error[:,:,:,2]<=0))
    #3rd case is where the order k2,k3,k1 has the lowest error
    case3 = np.where((error[:,:,:,2]-error[:,:,:,0]<=0)
                     * (error[:,:,:,2]-error[:,:,:,1] <=0))
    #we need case 3 for the eigenvectors
    
    #eigenvalue switching
    #wherever case1,case2 is true, we have to switch the order
    #back. In case 3, nothing has to happen.
    #for case 2
    temp[case2] = jbev1[case2] #current order is k2,k3,k1
    jbev1[case2] = jbev3[case2] #k1,k3,k1
    jbev3[case2] = temp[case2] #k1,k2,k2
    
    #for case 1
    temp[case1] = jbev1[case1] #current order is k2,k3,k1
    jbev1[case1] = jbev3[case1] #k1,k3,k1
    jbev3[case1] = temp[case1] #k1,k3,k2
    temp[case1] = jbev2[case1]
    jbev2[case1] = jbev3[case1] #k1,k2,k2
    jbev3[case1] = temp[case1] #k1,k2,k3
    
    #eigenvector switching
    #wherever case2,case3 is true, we have to switch the order
    #As the eigenvectors weren't rearranged yet, in case 1 nothing
    #happens.
    #for case 2 (we want k1,k3,k2)
    tempvec[case2] = jbevc2[case2] #current order is k1,k2,k3
    jbevc2[case2] = jbevc3[case2] #k1,k3,k3
    jbevc3[case2] = tempvec[case2] #k1,k3,k2
    
    #for case 3 (we want k2,k3,k1)
    tempvec[case3] = jbevc2[case3] #current order is k1,k2,k3
    jbevc2[case3] = jbevc3[case3] #k1,k3,k3
    jbevc3[case3] = tempvec[case3] #k1,k3,k2
    tempvec[case3] = jbevc1[case3] 
    jbevc1[case3] = jbevc3[case3] #k2,k3,k2
    jbevc3[case3] = tempvec[case3] #k2,k3,k1
    
    OGD = OGD[stencil//2:ogd0-stencil//2, stencil//2:ogd1-stencil//2,
              stencil//2:ogd2-stencil//2]
    
    return OGD, jbev1, jbev2, jbev3, jbevc1, jbevc2, jbevc3


def cepca_pointcloud(inp, rho, kr=3, order=2, cm=1, ocg="str", taulow=0,
                     mask=0, debug=0):                
    """
    Estimate curvature of an object, given as a pointcloud, using integral 
    invariants and pca. Returns two principal curvatures, two principal 
    directions, and the surface normal. Optionally also returns the occupancy 
    grid.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         coordinates x y z or a string that leads to the file that is in x y z
         format with no header. 
        -'rho' controls the amount of cells in the occupancy grid (=rho+1).
        -'kr' is the kernel radius.
        -'cm' stands for convolution mode.If ==1, the fft with zero-padding is
         used. If cm<1, then the discrete convolution with a certain kind of
         padding is used. If cm==0, zero-padding, if cm==0.25, 'reflect', 
         if cm==0.50, 'nearest', if cm==0.75, 'mirror', if cm==0.95, 'wrap'.
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >thr it is used in the relaxed occgrid and ignored otherwise.
         This exists to limit the error that can happen due to the usage of
         a relaxed occupancy grid.
        -'order' is the parameter that specifies the order of the eigenvalues.
         If order==0, then the order is not changed at all. 
         If order==1, then the eigenvalues and eigenvectors are ordered 
         according to the values of the eigenvalues. I.e. the biggest 
         eigenvalue is first, 2nd biggest is 2nd, etc etc.
         If order==2, then we use the error approximations of Pottmann07 to
         estimate which eigenvalues are the "first" and "second" eigenvalues
         that are needed to calculate the principal curvatures.
         If order ==3 and cm==1.5 then the reflect padding is used in the
         central difference computation. If order==3,cm==1, then zero padding.
         In all other cases for order==3, the same padding as in the
         convolution in the separate cm modes is used.
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'mask' specifies whether the occupancy grid shall be output or not
    
    Uses a ball kernel for the neighborhood intersection.
    Returns the two principal curvatures 'kappa1', 'kappa2' and the two
    principal directions jbevc1, jbevc2, and the surface normal jbevc3 for 
    every point on the occupancy grid. These vectores can be wrong in areas
    where kappa1==kappa2.
    """
    if debug==1:
        starttime = time.time()  
    
    if ocg=="str":
        OGD, OGB, xcoords, \
        ycoords, zcoords = og.constructpcagrids_pointcloud(inp, rho=rho, kr=kr,
                                                        ocg="str",
                                                        variant=1, debug=0)
    else:
        OGD, OGB, xcoords, \
        ycoords, zcoords = og.constructpcagrids_pointcloud(inp, rho=rho, kr=kr,
                                                        ocg=ocg, taulow=taulow, 
                                                        variant=1, debug=0)
    
    
    
    if debug==1:
        print("Got all the Occupancy Grids.\n"
              +"Shape of Domain OccGrid: " + str(np.shape(OGD)) + ".\n"
              + "Current Runtime: " + str(time.time() - starttime))

    """
    To calculate PCA in ball neighborhoods, we have to calculate 3 integrals:
        1) Vb = 1D \ast 1B, where 1D=OGD, 1B=OGB,
            
        2) sb = 1/Vb * (sb1, sb2, sb3)^T = 1/Vb * (1D \ast (-X*1B)
                                            
                       [1D \ast -x*1B]
              = 1/Vb * [1D \ast -y*1B]
                       [1D \ast -z*1B] , 
           where X=(x,y,z)^T, ogbx=-x*1B, ..., sbhelper1= 1D \ast -x*1B,
           sbhelper2=...
                                           
        3) Jb = (1D \ast (1B*X*X^T))- Vb*sb*sb^T
                                                   
                [1D \ast x^2*1B, 1D \ast xy*1B,   1D \ast xz*1B]
              = [1D \ast xy*1B,  1D \ast y^2*1B,  1D \ast yz*1B]
                [1D \ast xz*1B,  1D \ast yz*1B,  1D \ast z^2*1B]
                
                              [sb1^2,   sb1*sb2, sb1*sb3]
                - Vb * 1/Vb^2 [sb1*sb2, sb2^2,   sb1*sb3]
                              [sb1*sb3, sb1*sb3, sb3^2  ],
            where Jbhxx=1D \ast x^2*1B, Jbhxy=1D \ast xy*1B, ..., ogbxx=x^2*1B,
            ogbxy=xy*1B, ...
    """
    if cm<1:
        Jb, sbsbt, Vb, rmv, keep = cepca_discreteconvolution(OGD, OGB, xcoords,
                                                             ycoords, zcoords,
                                                             cm, debug=0)
        
        if debug==1:
            print("Calculated Jbhelper and sbsbt.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
    else:
        Jb, sbsbt, Vb, rmv, keep = cepca_fft(OGD, OGB, xcoords, ycoords, 
                                             zcoords, cm, debug=0)
        
        if debug==1:
            print("Calculated Jbhelper and sbsbt with FFT.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
    
    #set the values where we'd divide by zero to zero
    sbsbt[:,:,rmv[0],rmv[1],rmv[2]]=0
    
    #sbsbt[:,:,keep[0],keep[1],keep[2]]/=Vb[keep[0],keep[1],keep[2]] 
    # the above would work if I use // instead of /
    
    #and for the nonzero values, calculate 1/Vb * sbsbt
    sbsbt[:,:,keep[0],keep[1],keep[2]] = np.divide(sbsbt[:,:,keep[0],keep[1],
                                                          keep[2]],
                                                    Vb[keep[0],keep[1],keep[2]])
    
    #calculate Jb = Jbhelper - 1/Vb * sbsbt w/o using 
    #np.reshape and dividing by zero (thank you Virginie for the trick)
    Jb -= sbsbt
    
    # Jb is 3x3xRest and for np.linalg.eigh we need a structure
    # of Restx3x3
    Jb = np.transpose(Jb, (2, 3, 4, 0, 1))
            
    #calculate eigenvalues, where eigvals contains the eigvalues in
    #ascending order.
    eigvals, eigvects = np.linalg.eigh(Jb)
    
    if debug==1:
            print("Calculated all eigvals (and eigvects).\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    jbev1 = eigvals[:,:,:,0]
    jbev2 = eigvals[:,:,:,1]
    jbev3 = eigvals[:,:,:,2]
    
    jbevc1 = eigvects[:,:,:,0,:]
    jbevc2 = eigvects[:,:,:,1,:]
    jbevc3 = eigvects[:,:,:,2,:]
    
    #order eigenvalues and eigenvectors
    #order==0 means no ordering, order==1 means bigger eigenvalues are
    #first, order==2 uses the error approximations from Pottmann07.
    if order==1:
        jbev1, jbev2, jbev3,\
        jbevc1, jbevc2, jbevc3 = cepca_orderevals_size(jbev1, jbev2, jbev3,
                                                       jbevc1, jbevc2, jbevc3)
        
    elif order==2:
        jbev1, jbev2, jbev3,\
        jbevc1, jbevc2, jbevc3 = cepca_orderevals_error(kr,jbev1, jbev2, jbev3,
                                                        jbevc1, jbevc2, jbevc3)
              
    elif order==3:
        OGD,jbev1, jbev2, jbev3,\
        jbevc1, jbevc2, jbevc3 = cepca_orderevals_cdiffnrmls(OGD, cm, jbev1, 
                                                             jbev2, jbev3, 
                                                             jbevc1, jbevc2,
                                                             jbevc3)
        
    elif order==4:
        OGD,jbev1, jbev2, jbev3,\
        jbevc1, jbevc2, jbevc3 = cepca_orderevals_errnrmlsmix(OGD, cm, kr,
                                                              jbev1, jbev2, 
                                                              jbev3, jbevc1,
                                                              jbevc2, jbevc3)
            
    #this ends the ordering stuff!
    
    #now calculate principal curvatures
    kappa1 = 6/(np.pi*kr**6)*(jbev2-3*jbev1)+8/(5*kr)
    kappa2 = 6/(np.pi*kr**6)*(jbev1-3*jbev2)+8/(5*kr)
    
    #now output the results!
    
    if debug==1:
            print("Success!.\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    if debug==0:
        if mask==0:
            return kappa1, kappa2, jbevc1, jbevc2, jbevc3
        else:
            return kappa1, kappa2, jbevc1, jbevc2, jbevc3, OGD
    #two principal curvatures, two principal directions, one surface normal
    else:
        print("Debug: Returning kappa1, kappa2, jbev1, jbev2, jbev3, jbevc1,\
              jbevc2, jbevc3. Good luck!")
        return kappa1, kappa2, jbev1, jbev2, jbev3, jbevc1, jbevc2, jbevc3
    
def cepca_ocg(inpoccgrid, kr=3, order=2, cm=1, debug=0):
    """
    Estimate curvature of an object, given as an occupancy grid, using integral 
    invariants and pca. Returns two principal curvatures, two principal 
    directions, and the surface normal.
    
    Input:
        -'inp' is a strict or relaxed occupancy grid
        -'kr' is the kernel radius.
        -'cm' stands for convolution mode.If ==1, the fft with zero-padding is
         used. If cm<1, then the discrete convolution with a certain kind of
         padding is used. If cm==0, zero-padding, if cm==0.25, 'reflect', 
         if cm==0.50, 'nearest', if cm==0.75, 'mirror', if cm==0.95, 'wrap'.
        -'order' is the parameter that specifies the order of the eigenvalues.
         If order==0, then the order is not changed at all. 
         If order==1, then the eigenvalues and eigenvectors are ordered 
         according to the values of the eigenvalues. I.e. the biggest 
         eigenvalue is first, 2nd biggest is 2nd, etc etc.
         If order==2, then we use the error approximations of Pottmann07 to
         estimate which eigenvalues are the "first" and "second" eigenvalues
         that are needed to calculate the principal curvatures.
         If order ==3 and cm==1.5 then the reflect padding is used in the
         central difference computation. If order==3,cm==1, then zero padding.
         In all other cases for order==3, the same padding as in the
         convolution in the separate cm modes is used.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Uses a ball kernel for the neighborhood intersection.
    Returns the two principal curvatures 'kappa1', 'kappa2' and the two
    principal directions jbevc1, jbevc2, and the surface normal jbevc3 for 
    every point on the occupancy grid. These vectores can be wrong in areas
    where kappa1==kappa2.
    """
    if debug==1:
        starttime = time.time()  
        
    OGD, OGB, xcoords, \
    ycoords, zcoords = og.constructpcagrids_ocg(inpoccgrid, kr=kr, variant=1,
                                                debug=0)
    
    if debug==1:
        print("Got all the Occupancy Grids.\n"
              +"Shape of Domain OccGrid: " + str(np.shape(OGD)) + ".\n"
              + "Current Runtime: " + str(time.time() - starttime))

    """
    To calculate PCA in ball neighborhoods, we have to calculate 3 integrals:
        1) Vb = 1D \ast 1B, where 1D=OGD, 1B=OGB,
            
        2) sb = 1/Vb * (sb1, sb2, sb3)^T = 1/Vb * (1D \ast (-X*1B)
                                            
                       [1D \ast -x*1B]
              = 1/Vb * [1D \ast -y*1B]
                       [1D \ast -z*1B] , 
           where X=(x,y,z)^T, ogbx=-x*1B, ..., sbhelper1= 1D \ast -x*1B,
           sbhelper2=...
                                           
        3) Jb = (1D \ast (1B*X*X^T))- Vb*sb*sb^T
                                                   
                [1D \ast x^2*1B, 1D \ast xy*1B,   1D \ast xz*1B]
              = [1D \ast xy*1B,  1D \ast y^2*1B,  1D \ast yz*1B]
                [1D \ast xz*1B,  1D \ast yz*1B,  1D \ast z^2*1B]
                
                              [sb1^2,   sb1*sb2, sb1*sb3]
                - Vb * 1/Vb^2 [sb1*sb2, sb2^2,   sb1*sb3]
                              [sb1*sb3, sb1*sb3, sb3^2  ],
            where Jbhxx=1D \ast x^2*1B, Jbhxy=1D \ast xy*1B, ..., ogbxx=x^2*1B,
            ogbxy=xy*1B, ...
    """
    if cm<1:
        Jb, sbsbt, Vb, rmv, keep = cepca_discreteconvolution(OGD, OGB, xcoords,
                                                             ycoords, zcoords,
                                                             cm, debug=0)
        
        if debug==1:
            print("Calculated Jbhelper and sbsbt.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
    else:
        Jb, sbsbt, Vb, rmv, keep = cepca_fft(OGD, OGB, xcoords, ycoords, 
                                             zcoords, cm, debug=0)
        
        if debug==1:
            print("Calculated Jbhelper and sbsbt with FFT.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
    
    #set the values where we'd divide by zero to zero
    sbsbt[:,:,rmv[0],rmv[1],rmv[2]]=0
    
    #sbsbt[:,:,keep[0],keep[1],keep[2]]/=Vb[keep[0],keep[1],keep[2]] 
    # the above would work if I use // instead of /
    
    #and for the nonzero values, calculate 1/Vb * sbsbt
    sbsbt[:,:,keep[0],keep[1],keep[2]] = np.divide(sbsbt[:,:,keep[0],keep[1],
                                                          keep[2]],
                                                    Vb[keep[0],keep[1],keep[2]])
    
    #calculate Jb = Jbhelper - 1/Vb * sbsbt w/o using 
    #np.reshape and dividing by zero (thank you Virginie for the trick)
    Jb -= sbsbt
    
    # Jb is 3x3xRest and for np.linalg.eigh we need a structure
    # of Restx3x3
    Jb = np.transpose(Jb, (2, 3, 4, 0, 1))
            
    #calculate eigenvalues, where eigvals contains the eigvalues in
    #ascending order.
    eigvals, eigvects = np.linalg.eigh(Jb)
    
    if debug==1:
            print("Calculated all eigvals (and eigvects).\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    jbev1 = eigvals[:,:,:,0]
    jbev2 = eigvals[:,:,:,1]
    jbev3 = eigvals[:,:,:,2]
    
    jbevc1 = eigvects[:,:,:,0,:]
    jbevc2 = eigvects[:,:,:,1,:]
    jbevc3 = eigvects[:,:,:,2,:]
    
    #order eigenvalues and eigenvectors
    #order==0 means no ordering, order==1 means bigger eigenvalues are
    #first, order==2 uses the error approximations from Pottmann07.
    if order==1:
        jbev1, jbev2, jbev3,\
        jbevc1, jbevc2, jbevc3 = cepca_orderevals_size(jbev1, jbev2, jbev3,
                                                       jbevc1, jbevc2, jbevc3)
        
    elif order==2:
        jbev1, jbev2, jbev3,\
        jbevc1, jbevc2, jbevc3 = cepca_orderevals_error(kr,jbev1, jbev2, jbev3,
                                                        jbevc1, jbevc2, jbevc3)
              
    elif order==3:
        OGD,jbev1, jbev2, jbev3,\
        jbevc1, jbevc2, jbevc3 = cepca_orderevals_cdiffnrmls(OGD, cm, jbev1, 
                                                             jbev2, jbev3, 
                                                             jbevc1, jbevc2,
                                                             jbevc3)
        
    elif order==4:
        OGD,jbev1, jbev2, jbev3,\
        jbevc1, jbevc2, jbevc3 = cepca_orderevals_errnrmlsmix(OGD, cm, kr,
                                                              jbev1, jbev2, 
                                                              jbev3, jbevc1,
                                                              jbevc2, jbevc3)
            
    #this ends the ordering stuff!
    
    #now calculate principal curvatures
    kappa1 = 6/(np.pi*kr**6)*(jbev2-3*jbev1)+8/(5*kr)
    kappa2 = 6/(np.pi*kr**6)*(jbev1-3*jbev2)+8/(5*kr)
    
    #now output the results!
    
    if debug==1:
            print("Success!.\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    if debug==0:
        return kappa1, kappa2, jbevc1, jbevc2, jbevc3
    #two principal curvatures, two principal directions, one surface normal
    else:
        print("Debug: Returning kappa1, kappa2, jbev1, jbev2, jbev3, jbevc1,\
              jbevc2, jbevc3. Good luck!")
        return kappa1, kappa2, jbev1, jbev2, jbev3, jbevc1, jbevc2, jbevc3
   

def cepca_msavg_pointcloud(inp, rho, startscale=3, endscale=12, scaledist=3, 
                           ocg="str", taulow=0, order=2, cm=1, debug=0):
    """
    Curvature estimation on a pointcloud using integral invariants, pca,
    and a multiscale averaging method.
    Returns two principal curvatures, two principal 
    directions, and the surface normal.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         coordinates x y z or a string that leads to the file that is in x y z
         format with no header. 
        -'rho' controls the amount of cells in the occupancy grid (=rho+1).
        -'cm' stands for convolution mode.If ==1, the fft with zero-padding is
         used. If cm<1, then the discrete convolution with a certain kind of
         padding is used. If cm==0, zero-padding, if cm==0.25, 'reflect', 
         if cm==0.50, 'nearest', if cm==0.75, 'mirror', if cm==0.95, 'wrap'.
        -'order' is the parameter that specifies the order of the eigenvalues.
         If order==0, then the order is not changed at all. 
         If order==1, then the eigenvalues and eigenvectors are ordered 
         according to the values of the eigenvalues. I.e. the biggest 
         eigenvalue is first, 2nd biggest is 2nd, etc etc.
         If order==2, then we use the error approximations of Pottmann07 to
         estimate which eigenvalues are the "first" and "second" eigenvalues
         that are needed to calculate the principal curvatures.
         If order ==3 and cm==1.5 then the reflect padding is used in the
         central difference computation. If order==3,cm==1, then zero padding.
         In all other cases for order==3, the same padding as in the
         convolution in the separate cm modes is used.
        -multiscale parameters: instead of calculating the principal on one 
         kernel radius kr, we calculate them over multiple kernel radii and
         average the results. We start at kr 'startscale', end at 'endscale',
         and specify the distance between each other using 'scaledist'.
         E.g., startscale=3, endscale=12, scaledist=3, then we average over
         the scales 3,6,9,12. If scaledist is 4 and rest is same, then 3,7,11.
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the
         usage of a relaxed occupancy grid.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Uses a ball kernel for the neighborhood intersection.
    Returns the two principal curvatures 'kappa1', 'kappa2' and the two
    principal directions jbevc1, jbevc2, and the surface normal jbevc3 for 
    every point on the occupancy grid. These vectors can be wrong in areas
    wherever kappa1==kappa2 and in areas of visual artifacts.
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
        OGD = og.constructoccgrid_pointcloud([z,y,x], rho)
    else:
        OGD = og.constructoccgrid_pointcloud([z,y,x, vals], rho, ocg=ocg, 
                                             taulow=taulow)
    
    #get amounts of elemts in range to properly induce weights
    weights = 0
    for i in range(startscale, endscale+1, scaledist):
        weights = weights+1 
        #yes, this is hacky, but it works and is still fast enough
        #I can't come up with a better, more elegant solution right now
    
    weights = 1/weights
    
    #init averaged parameters
    pc1 = np.zeros(np.shape(OGD))
    pc2 = np.zeros(np.shape(OGD))
    k3 = np.zeros(np.shape(OGD))
    eigv1 = np.zeros((np.shape(OGD)+(3,)))
    eigv2 = np.zeros((np.shape(OGD)+(3,)))
    eigv3 = np.zeros((np.shape(OGD)+(3,)))
    
    #init eigvals and eigvects of jb
    jbev1 = np.zeros(np.shape(OGD))
    jbev2 = np.zeros(np.shape(OGD))
    jbev3 = np.zeros(np.shape(OGD))
    jbevc1 = np.zeros((np.shape(OGD)+(3,)))
    jbevc2 = np.zeros((np.shape(OGD)+(3,)))
    jbevc3 = np.zeros((np.shape(OGD)+(3,))) 
    
    for kr in range(startscale, endscale+1, scaledist):
        OGD, OGB, xcoords,\
        ycoords, zcoords = og.constructpcagrids_ms_pointcloud([z,y,x], kr, rho,
                                                         startscale,
                                                         scaledist, ocg=ocg,
                                                         taulow=taulow)
    
        if debug==1:
            print("Got all the Occupancy Grids.\n"
                  +"Shape of Domain OccGrid: " + str(np.shape(OGD)) + ".\n"
                  + "Current Runtime: " + str(time.time() - starttime))
    
        """
        To calculate PCA in ball neighborhoods, we have to calculate 3 integrals:
            1) Vb = 1D \ast 1B, where 1D=OGD, 1B=OGB,
                
            2) sb = 1/Vb * (sb1, sb2, sb3)^T = 1/Vb * (1D \ast (-X*1B)
                                                
                           [1D \ast -x*1B]
                  = 1/Vb * [1D \ast -y*1B]
                           [1D \ast -z*1B] , 
               where X=(x,y,z)^T, ogbx=-x*1B, ..., sbhelper1= 1D \ast -x*1B,
               sbhelper2=...
                                               
            3) Jb = (1D \ast (1B*X*X^T))- Vb*sb*sb^T
                                                       
                    [1D \ast x^2*1B, 1D \ast xy*1B,   1D \ast xz*1B]
                  = [1D \ast xy*1B,  1D \ast y^2*1B,  1D \ast yz*1B]
                    [1D \ast xz*1B,  1D \ast yz*1B,  1D \ast z^2*1B]
                    
                                  [sb1^2,   sb1*sb2, sb1*sb3]
                    - Vb * 1/Vb^2 [sb1*sb2, sb2^2,   sb1*sb3]
                                  [sb1*sb3, sb1*sb3, sb3^2  ],
                where Jbhxx=1D \ast x^2*1B, Jbhxy=1D \ast xy*1B, ..., ogbxx=x^2*1B,
                ogbxy=xy*1B, ...
        """
        if cm<1:
            Jb, sbsbt, Vb, rmv, keep = cepca_discreteconvolution(OGD, OGB, xcoords,
                                                             ycoords, zcoords,
                                                             cm, debug=0)
            
            if debug==1:
                print("Calculated Jbhelper and sbsbt.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
        else:
            Jb, sbsbt, Vb, rmv, keep = cepca_fft(OGD, OGB, xcoords, ycoords, 
                                             zcoords, cm, debug=0)
            
            if debug==1:
                print("Calculated Jbhelper and sbsbt with FFT.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
                
        
        
        #set the values where we'd divide by zero to zero
        sbsbt[:,:,rmv[0],rmv[1],rmv[2]]=0
        
        #and for the nonzero values, calculate 1/Vb * sbsbt
        sbsbt[:,:,keep[0],keep[1],keep[2]] = np.divide(sbsbt[:,:,keep[0],keep[1],
                                                              keep[2]],
                                                        Vb[keep[0],keep[1],keep[2]])
        
        
        #calculate Jb = Jbhelper - 1/Vb * sbsbt w/o using 
        #np.reshape and dividing by zero (thank you Virginie for the trick)
        Jb -= sbsbt
        
        # Jb is 3x3xRest and for np.linalg.eigh we need a structure
        # of Restx3x3
        Jb = np.transpose(Jb, (2, 3, 4, 0, 1))
        
        #calculate eigenvalues, where eigvals contains the eigvalues in
        #ascending order.
        eigvals, eigvects = np.linalg.eigh(Jb)
        
        if debug==1:
                print("Calculated all eigvals (and eigvects).\n"
                      + "Current Runtime: " + str(time.time() - starttime))
    
        if kr==startscale:
            jbev1 = eigvals[:,:,:,0]
            jbev2 = eigvals[:,:,:,1]
            jbev3 = eigvals[:,:,:,2]
            
            jbevc1 = eigvects[:,:,:,0,:]
            jbevc2 = eigvects[:,:,:,1,:]
            jbevc3 = eigvects[:,:,:,2,:]
        else:
            jbev1prev = jbev1
            jbev2prev = jbev2
            jbev3prev = jbev3
            
            jbevc1prev = jbevc1
            jbevc2prev = jbevc2
            jbevc3prev = jbevc3
            
            jbev1 = eigvals[:,:,:,0]
            jbev2 = eigvals[:,:,:,1]
            jbev3 = eigvals[:,:,:,2]
            
            jbevc1 = eigvects[:,:,:,0,:]
            jbevc2 = eigvects[:,:,:,1,:]
            jbevc3 = eigvects[:,:,:,2,:]
    
        
    
        #order eigenvalues and eigenvectors
        #order==0 means no ordering, order==1 means bigger eigenvalues are
        #first, order==2 uses the error approximations from Pottmann07.
        if order==1:
            jbev1, jbev2, jbev3,\
            jbevc1, jbevc2, jbevc3 = cepca_orderevals_size(jbev1, jbev2, jbev3,
                                                       jbevc1, jbevc2, jbevc3)
            
        elif order==2:
            jbev1, jbev2, jbev3,\
            jbevc1, jbevc2, jbevc3 = cepca_orderevals_error(kr, jbev1, jbev2,
                                                            jbev3, jbevc1,
                                                            jbevc2, jbevc3)
                  
        elif order==3:
            OGD,jbev1, jbev2, jbev3,\
            jbevc1, jbevc2, jbevc3 = cepca_orderevals_cdiffnrmls(OGD, cm, jbev1, 
                                                                 jbev2, jbev3, 
                                                                 jbevc1, jbevc2,
                                                                 jbevc3)
            
        elif order==4:
            OGD,jbev1, jbev2, jbev3,\
            jbevc1, jbevc2, jbevc3 = cepca_orderevals_errnrmlsmix(OGD, cm, kr,
                                                                  jbev1, jbev2, 
                                                                  jbev3, jbevc1,
                                                                  jbevc2, jbevc3)
                
        #this ends the ordering stuff!
        
        #now set the split kernel stuff back together
        if kr!=startscale:
            jbev1 = jbev1 + jbev1prev
            jbev2 = jbev2 + jbev2prev
            jbev3 = jbev3 + jbev3prev
            
            jbevc1 = jbevc1 + jbevc1prev
            jbevc2 = jbevc2 + jbevc2prev
            jbevc3 = jbevc3 + jbevc3prev
            #this is a bit of a stretch, but I found a note about fixing an 
            #error in this method and I'm not sure what it was referring to
            #exactly. Comparing this method to my pseudoalgorithm, there's a
            #difference in how im calculating and weighing the principal
            #curvatures and eigenvalues, so maybe maybe, this way of putting
            #together the eigvals and then the principal curvatures is wrong,
            #but I can't tell right now. As far as I'm concerned it looks
            #and works alright.
            
        
        
        #now calculate principal curvatures for this scale
        fstpc = 6/(np.pi*kr**6)*(jbev2-3*jbev1)+8/(5*kr)
        sndpc = 6/(np.pi*kr**6)*(jbev1-3*jbev2)+8/(5*kr)
        
        #now average the results
        pc1 = pc1 + weights*fstpc
        pc2 = pc2 + weights*sndpc
        k3 = k3 + weights*jbev3
        eigv1 = eigv1 + weights*jbevc1
        eigv2 = eigv2 + weights*jbevc2
        eigv3 = eigv3 + weights*jbevc3
    
    #averaging and pca is done, output results.
    
    if debug==1:
            print("Success!.\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    if debug==0:
        return pc1, pc2, eigv1, eigv2, eigv3 #two principal curvatures,
                                      #one surface normal
    else:
        print("Debug: Returning pc1, pc2, k3, eigv1, eigv2, eigv3.Good luck!\n"
              + "Current Runtime: " + str(time.time() - starttime))
        
        return pc1, pc2, k3, eigv1, eigv2, eigv3

def cepca_msavg_ocg(inpocg, startscale=3, endscale=12, scaledist=3, 
                    order=2, cm=1, debug=0):
    """
    Curvature estimation on an occupancy grid using integral invariants, pca,
    and a multiscale averaging method.
    Returns two principal curvatures, two principal 
    directions, and the surface normal.
    
    Input:
        -'inp' is a strict or relaxed occupancy grid.
        -'cm' stands for convolution mode.If ==1, the fft with zero-padding is
         used. If cm<1, then the discrete convolution with a certain kind of
         padding is used. If cm==0, zero-padding, if cm==0.25, 'reflect', 
         if cm==0.50, 'nearest', if cm==0.75, 'mirror', if cm==0.95, 'wrap'.
        -'order' is the parameter that specifies the order of the eigenvalues.
         If order==0, then the order is not changed at all. 
         If order==1, then the eigenvalues and eigenvectors are ordered 
         according to the values of the eigenvalues. I.e. the biggest 
         eigenvalue is first, 2nd biggest is 2nd, etc etc.
         If order==2, then we use the error approximations of Pottmann07 to
         estimate which eigenvalues are the "first" and "second" eigenvalues
         that are needed to calculate the principal curvatures.
         If order ==3 and cm==1.5 then the reflect padding is used in the
         central difference computation. If order==3,cm==1, then zero padding.
         In all other cases for order==3, the same padding as in the
         convolution in the separate cm modes is used.
        -multiscale parameters: instead of calculating the principal on one 
         kernel radius kr, we calculate them over multiple kernel radii and
         average the results. We start at kr 'startscale', end at 'endscale',
         and specify the distance between each other using 'scaledist'.
         E.g., startscale=3, endscale=12, scaledist=3, then we average over
         the scales 3,6,9,12. If scaledist is 4 and rest is same, then 3,7,11.
        
    
    Uses a ball kernel for the neighborhood intersection.
    Returns the two principal curvatures 'kappa1', 'kappa2' and the two
    principal directions jbevc1, jbevc2, and the surface normal jbevc3 for 
    every point on the occupancy grid. These vectores can be wrong in areas
    where kappa1==kappa2.
    """
    if debug==1:
        starttime = time.time()  
    
    #get amounts of elemts in range to properly induce weights
    weights = 0
    for i in range(startscale, endscale+1, scaledist):
        weights = weights+1 
        #yes, this is hacky, but it works and is still fast enough
        #I can't come up with a better, more elegant solution right now
    
    weights = 1/weights
    
    #init averaged parameters
    pc1 = np.zeros(np.shape(inpocg))
    pc2 = np.zeros(np.shape(inpocg))
    k3 = np.zeros(np.shape(inpocg))
    eigv1 = np.zeros((np.shape(inpocg)+(3,)))
    eigv2 = np.zeros((np.shape(inpocg)+(3,)))
    eigv3 = np.zeros((np.shape(inpocg)+(3,)))
    
    #init eigvals and eigvects of jb
    jbev1 = np.zeros(np.shape(inpocg))
    jbev2 = np.zeros(np.shape(inpocg))
    jbev3 = np.zeros(np.shape(inpocg))
    jbevc1 = np.zeros((np.shape(inpocg)+(3,)))
    jbevc2 = np.zeros((np.shape(inpocg)+(3,)))
    jbevc3 = np.zeros((np.shape(inpocg)+(3,)))
    
    for kr in range(startscale, endscale+1, scaledist):
        OGD, OGB, xcoords,\
        ycoords, zcoords = og.constructpcagrids_ms_ocg(inpocg, kr,
                                                       startscale,
                                                       scaledist)
        
        if debug==1:
            print("Got all the Occupancy Grids.\n"
                  +"Shape of Domain OccGrid: " + str(np.shape(OGD)) + ".\n"
                  + "Current Runtime: " + str(time.time() - starttime))
    
        """
        To calculate PCA in ball neighborhoods, we have to calculate 3 integrals:
            1) Vb = 1D \ast 1B, where 1D=OGD, 1B=OGB,
                
            2) sb = 1/Vb * (sb1, sb2, sb3)^T = 1/Vb * (1D \ast (-X*1B)
                                                
                           [1D \ast -x*1B]
                  = 1/Vb * [1D \ast -y*1B]
                           [1D \ast -z*1B] , 
               where X=(x,y,z)^T, ogbx=-x*1B, ..., sbhelper1= 1D \ast -x*1B,
               sbhelper2=...
                                               
            3) Jb = (1D \ast (1B*X*X^T))- Vb*sb*sb^T
                                                       
                    [1D \ast x^2*1B, 1D \ast xy*1B,   1D \ast xz*1B]
                  = [1D \ast xy*1B,  1D \ast y^2*1B,  1D \ast yz*1B]
                    [1D \ast xz*1B,  1D \ast yz*1B,  1D \ast z^2*1B]
                    
                                  [sb1^2,   sb1*sb2, sb1*sb3]
                    - Vb * 1/Vb^2 [sb1*sb2, sb2^2,   sb1*sb3]
                                  [sb1*sb3, sb1*sb3, sb3^2  ],
                where Jbhxx=1D \ast x^2*1B, Jbhxy=1D \ast xy*1B, ..., ogbxx=x^2*1B,
                ogbxy=xy*1B, ...
        """
        if cm<1:
            Jb, sbsbt, Vb, rmv, keep = cepca_discreteconvolution(OGD, OGB, xcoords,
                                                             ycoords, zcoords,
                                                             cm, debug=0)
            
            if debug==1:
                print("Calculated Jbhelper and sbsbt.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
        else:
            Jb, sbsbt, Vb, rmv, keep = cepca_fft(OGD, OGB, xcoords, ycoords, 
                                             zcoords, cm, debug=0)
            
            if debug==1:
                print("Calculated Jbhelper and sbsbt with FFT.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
                
        
        
        #set the values where we'd divide by zero to zero
        sbsbt[:,:,rmv[0],rmv[1],rmv[2]]=0
        
        #and for the nonzero values, calculate 1/Vb * sbsbt
        sbsbt[:,:,keep[0],keep[1],keep[2]] = np.divide(sbsbt[:,:,keep[0],keep[1],
                                                              keep[2]],
                                                        Vb[keep[0],keep[1],keep[2]])
        
        
        #calculate Jb = Jbhelper - 1/Vb * sbsbt w/o using 
        #np.reshape and dividing by zero (thank you Virginie for the trick)
        Jb -= sbsbt
        
        # Jb is 3x3xRest and for np.linalg.eigh we need a structure
        # of Restx3x3
        Jb = np.transpose(Jb, (2, 3, 4, 0, 1))
        
        #calculate eigenvalues, where eigvals contains the eigvalues in
        #ascending order.
        eigvals, eigvects = np.linalg.eigh(Jb)
        
        if debug==1:
                print("Calculated all eigvals (and eigvects).\n"
                      + "Current Runtime: " + str(time.time() - starttime))
    
        if kr==startscale:
            jbev1 = eigvals[:,:,:,0]
            jbev2 = eigvals[:,:,:,1]
            jbev3 = eigvals[:,:,:,2]
            
            jbevc1 = eigvects[:,:,:,0,:]
            jbevc2 = eigvects[:,:,:,1,:]
            jbevc3 = eigvects[:,:,:,2,:]
        else:
            jbev1prev = jbev1
            jbev2prev = jbev2
            jbev3prev = jbev3
            
            jbevc1prev = jbevc1
            jbevc2prev = jbevc2
            jbevc3prev = jbevc3
            
            jbev1 = eigvals[:,:,:,0]
            jbev2 = eigvals[:,:,:,1]
            jbev3 = eigvals[:,:,:,2]
            
            jbevc1 = eigvects[:,:,:,0,:]
            jbevc2 = eigvects[:,:,:,1,:]
            jbevc3 = eigvects[:,:,:,2,:]
    
        
    
        #order eigenvalues and eigenvectors
        #order==0 means no ordering, order==1 means bigger eigenvalues are
        #first, order==2 uses the error approximations from Pottmann07.
        if order==0.5:
            #this was added later on and directly uses the ascending order
            #resulting from "eigh" to order the eigvals descendingly.
            temp = eigvals[:,:,:,0]
            jbev1 = eigvals[:,:,:,2]
            jbev2 = eigvals[:,:,:,1]
            jbev3 = temp
            
            temp = eigvects[:,:,:,0]
            jbevc1 = eigvects[:,:,:,2,:]
            jbevc2 = eigvects[:,:,:,1,:]
            jbevc3 = temp
        
        elif order==1:
            jbev1, jbev2, jbev3,\
            jbevc1, jbevc2, jbevc3 = cepca_orderevals_size(jbev1, jbev2, jbev3,
                                                       jbevc1, jbevc2, jbevc3)
            
        elif order==2:
            jbev1, jbev2, jbev3,\
            jbevc1, jbevc2, jbevc3 = cepca_orderevals_error(kr, jbev1, jbev2,
                                                            jbev3, jbevc1,
                                                            jbevc2, jbevc3)
                  
        elif order==3:
            OGD,jbev1, jbev2, jbev3,\
            jbevc1, jbevc2, jbevc3 = cepca_orderevals_cdiffnrmls(OGD, cm, jbev1, 
                                                                 jbev2, jbev3, 
                                                                 jbevc1, jbevc2,
                                                                 jbevc3)
            
        elif order==4:
            OGD,jbev1, jbev2, jbev3,\
            jbevc1, jbevc2, jbevc3 = cepca_orderevals_errnrmlsmix(OGD, cm, kr,
                                                                  jbev1, jbev2, 
                                                                  jbev3, jbevc1,
                                                                  jbevc2, jbevc3)
                
        #this ends the ordering stuff!
        
        #now set the split kernel stuff back together
        if kr!=startscale:
            jbev1 = jbev1 + jbev1prev
            jbev2 = jbev2 + jbev2prev
            jbev3 = jbev3 + jbev3prev
            
            jbevc1 = jbevc1 + jbevc1prev
            jbevc2 = jbevc2 + jbevc2prev
            jbevc3 = jbevc3 + jbevc3prev
        
        #now calculate principal curvatures for this scale
        fstpc = 6/(np.pi*kr**6)*(jbev2-3*jbev1)+8/(5*kr)
        sndpc = 6/(np.pi*kr**6)*(jbev1-3*jbev2)+8/(5*kr)
        
        
        #now average the results
        pc1 = pc1 + weights*fstpc
        pc2 = pc2 + weights*sndpc
        k3 = k3 + weights*jbev3
        eigv1 = eigv1 + weights*jbevc1
        eigv2 = eigv2 + weights*jbevc2
        eigv3 = eigv3 + weights*jbevc3
    
    #averaging and pca is done, output results.
    
    if debug==1:
            print("Success!.\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    if debug==0:
        return pc1, pc2, eigv1, eigv2, eigv3 #two principal curvatures,
                                      #one surface normal
    else:
        print("Debug: Returning pc1, pc2, k3, eigv1, eigv2, eigv3.Good luck!\n"
              + "Current Runtime: " + str(time.time() - starttime))
        
        return pc1, pc2, k3, eigv1, eigv2, eigv3
    

def cemean_simple_pointcloud(inp, rho=1, kr=6, fft=1, ocg="str", taulow=0,
                             debug=0):
    """
    Calculates a simple mean curvature estimation of a pointcloud. Returns a
    matrix with the mean curvature value at each voxel.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         coordinates x y z or a string that leads to the file that is in x y z
         format with no header. 
        -'rho' controls the amount of cells in the occupancy grid (=rho+1).
        -'kr' is the kernel radius.
        -'fft' specifies wether the fft convolution shall be used (=1) 
         or not (=0).
         -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the
         usage of a relaxed occupancy grid.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Uses a ball kernel for the neighborhood intersection.
    Returns the mean curvature for each voxel.
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
    
    elif isinstance(inp,(list,np.ndarray)):
        z = inp[:,0]
        y = inp[:,1]
        x = inp[:,2]
        
        if ocg=="rlx":
            vals = inp[:,3]
    else:
        raise NameError('Input can be an already loaded pointcloud that \
                        consists of the three coordinates x y z or a string \
                        that leads to the file that is in x y z format with \
                        no header. ')
                        
    if debug==1:
        print("Initialised the input point cloud.\n"
              + "Current Runtime: " + str(time.time() - starttime))
        print("Number of Points in the pointcloud: " + str(np.shape(x)[0]))
        
    if ocg=="str":
        OGD = og.constructoccgrid_pointcloud([z, y, x], rho)
    else:
        OGD = og.constructoccgrid_pointcloud([z, y, x, vals], rho, ocg=ocg, 
                                             taulow=taulow)
    
    
    OGB = og.constructoccgrid_ball(kr)
            
    
    if debug==1:
        print("Got all the Occupancy Grids.\n"
              + "Current Runtime: " + str(time.time() - starttime))

    if fft==0:
        Vb = nd.filters.convolve(OGD, OGB, mode='constant', cval=0.0)
        
        if debug==1:
            print("Calculated Vb without FFT.\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    else:
        Vb = sg.fftconvolve(OGD, OGB, mode='same')
        
        if debug==1:
            print("Calculated Vb with FFT.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
    

    mc = (4/(np.pi*np.abs(kr)**4)) * (( (2*np.pi/3) * np.abs(kr)**3 - Vb))    
        
    if debug==1:
            print("Done.\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    if debug==0:
        return mc
    else:
        print("Debug: Returning OGD, OGN, Vb, mc. Good luck!")
        return OGD, OGB, Vb, mc
    
def cemean_simple_ocg(inp, kr=6, fft=1, debug=0):
    """
    Calculates a simple mean curvature estimation of an occupancy grid. 
    Returns a matrix with the mean curvature value at each voxel.
    
    Input:
        -'inp' is a strict or relaxed occupancy grid
        -'kr' is the kernel radius.
        -'fft' specifies wether the fft convolution shall be used (=1) 
         or not (=0).
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Uses a ball kernel for the neighborhood intersection.
    Returns the mean curvature for each voxel.
    """
    if debug==1:
        starttime = time.time()    
                        
    if debug==1:
        print("Initialised the input point cloud.\n"
              + "Current Runtime: " + str(time.time() - starttime))
    
    OGB = og.constructoccgrid_ball(kr)
            
    
    if debug==1:
        print("Got all the Occupancy Grids.\n"
              + "Current Runtime: " + str(time.time() - starttime))

    if fft==0:
        Vb = nd.filters.convolve(inp, OGB, mode='constant', cval=0.0)
        
        if debug==1:
            print("Calculated Vb without FFT.\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    else:
        Vb = sg.fftconvolve(inp, OGB, mode='same')
        
        if debug==1:
            print("Calculated Vb with FFT.\n"
                  + "Current Runtime: " + str(time.time() - starttime))
    

    mc = (4/(np.pi*np.abs(kr)**4)) * (( (2*np.pi/3) * np.abs(kr)**3 - Vb))    
        
    if debug==1:
            print("Done.\n"
                  + "Current Runtime: " + str(time.time() - starttime))

    if debug==0:
        return mc
    else:
        print("Debug: Returning OGD, OGN, Vb, mc. Good luck!")
        return inp, OGB, Vb, mc
    
def cemean_principalcurv(kappa1,kappa2, debug=0):
    """
    Calculates the mean curvature from two matrices that contain the first and
    second principal curvature, respectively. Returns a matrix with the mean 
    curvature at each voxel.
    
    Input:
        -kappa1,2 are the two principal curvature values and they are given
         as a matrix.
        -'debug' for debugging
    
    Returns mean curvature in a matrix.
    """
    if debug==1:
        starttime = time.time()    

    #calculate mean curvature
    mc = (kappa1 + kappa2)/2
    
        
    if debug==1:
                print("Success!.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
        
    return mc
    

def cemeanpca_pointcloud(inp, rho, kr=3, order=2, cm=1, ocg="str", taulow=0,
                         debug=0):
    """
    Calculates the mean curvature of a pointcloud using pca and integral
    invariants. Returns mean curvature, both principal directions and the
    surface normal.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         coordinates x y z or a string that leads to the file that is in x y z
         format with no header. 
        -'rho' controls the amount of cells in the occupancy grid (=rho+1).
        -'kr' is the kernel radius.
        -'cm' stands for convolution mode.If ==1, the fft with zero-padding is
         used. If cm<1, then the discrete convolution with a certain kind of
         padding is used. If cm==0, zero-padding, if cm==0.25, 'reflect', 
         if cm==0.50, 'nearest', if cm==0.75, 'mirror', if cm==0.95, 'wrap'.
        -'order' is the parameter that specifies the order of the eigenvalues.
         If order==0, then the order is not changed at all. 
         If order==1, then the eigenvalues and eigenvectors are ordered 
         according to the values of the eigenvalues. I.e. the biggest 
         eigenvalue is first, 2nd biggest is 2nd, etc etc.
         If order==2, then we use the error approximations of Pottmann07 to
         estimate which eigenvalues are the "first" and "second" eigenvalues
         that are needed to calculate the principal curvatures.
         If order ==3 and cm==1.5 then the reflect padding is used in the
         central difference computation. If order==3,cm==1, then zero padding.
         In all other cases for order==3, the same padding as in the
         convolution in the separate cm modes is used.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the
         usage of a relaxed occupancy grid.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Uses a ball kernel for the neighborhood intersection.
    Returns mean curvature, both principal directions and the
    surface normal each in their own matrices where each voxel corresponds to
    the value.
    """
    if debug==1:
        starttime = time.time()    
        
    kappa1, kappa2, pd1, pd2, sn = cepca_pointcloud(inp, rho, kr=kr, 
                                                    order=order, cm=cm, 
                                                    ocg=ocg, taulow=taulow)
    
    #calculate mean curvature
    mc = (kappa1 + kappa2)/2
    
        
    if debug==1:
                print("Success!.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
        
    return mc, pd1, pd2, sn

def cemeanpca_ocg(inp, kr=3, order=2, cm=1, debug=0):
    """
    Calculates the mean curvature of strict occupancy grid using pca & integral
    invariants. Returns mean curvature, both principal directions and the
    surface normal.
    
    Input:
        -'inp' is a strict or relaxed occipancy grid
        -'kr' is the kernel radius.
        -'cm' stands for convolution mode.If ==1, the fft with zero-padding is
         used. If cm<1, then the discrete convolution with a certain kind of
         padding is used. If cm==0, zero-padding, if cm==0.25, 'reflect', 
         if cm==0.50, 'nearest', if cm==0.75, 'mirror', if cm==0.95, 'wrap'.
        -'order' is the parameter that specifies the order of the eigenvalues.
         If order==0, then the order is not changed at all. 
         If order==1, then the eigenvalues and eigenvectors are ordered 
         according to the values of the eigenvalues. I.e. the biggest 
         eigenvalue is first, 2nd biggest is 2nd, etc etc.
         If order==2, then we use the error approximations of Pottmann07 to
         estimate which eigenvalues are the "first" and "second" eigenvalues
         that are needed to calculate the principal curvatures.
         If order ==3 and cm==1.5 then the reflect padding is used in the
         central difference computation. If order==3,cm==1, then zero padding.
         In all other cases for order==3, the same padding as in the
         convolution in the separate cm modes is used.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Uses a ball kernel for the neighborhood intersection.
    Returns mean curvature, both principal directions and the
    surface normal each in their own matrices where each voxel corresponds to
    the value.
    """
    if debug==1:
        starttime = time.time()    
        
    kappa1, kappa2, pd1, pd2, sn = cepca_ocg(inp, kr=kr, order=order, cm=cm)
    
    #calculate mean curvature
    mc = (kappa1 + kappa2)/2
    
        
    if debug==1:
                print("Success!.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
        
    return mc, pd1, pd2, sn
    

def cegauss_principalcurv(kappa1,kappa2, debug=0):
    """
    Calculates the gaussian curvature from two matrices that contain the first
    and second principal curvature, respectively. Returns a matrix with the 
    mean curvature at each voxel.
    
    Input:
        -kappa1,2 are the two principal curvature values and they are given
         as a matrix.
        -'debug' for debugging
    
    Returns gaussian curvature in a matrix.
    """
    if debug==1:
        starttime = time.time()    

    #calculate gaussian curvature
    gc = kappa1 * kappa2
    
        
    if debug==1:
                print("Success!.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
        
    return gc
    
def cegausspca_pointcloud(inp, rho, kr=3, order=2, cm=1, ocg="str", taulow=0,
                          debug=0):
    """
    Calculates gaussian curvature of a pointcloud using pca and integral
    invariants. Returns gaussian curvature, both principal directions and the
    surface normal.
    
    Input:
        -'inp' can be an already loaded pointcloud that consists of the three
         coordinates x y z or a string that leads to the file that is in x y z
         format with no header. 
        -'rho' controls the amount of cells in the occupancy grid (=rho+1).
        -'kr' is the kernel radius.
        -'cm' stands for convolution mode.If ==1, the fft with zero-padding is
         used. If cm<1, then the discrete convolution with a certain kind of
         padding is used. If cm==0, zero-padding, if cm==0.25, 'reflect', 
         if cm==0.50, 'nearest', if cm==0.75, 'mirror', if cm==0.95, 'wrap'.
        -'order' is the parameter that specifies the order of the eigenvalues.
         If order==0, then the order is not changed at all. 
         If order==1, then the eigenvalues and eigenvectors are ordered 
         according to the values of the eigenvalues. I.e. the biggest 
         eigenvalue is first, 2nd biggest is 2nd, etc etc.
         If order==2, then we use the error approximations of Pottmann07 to
         estimate which eigenvalues are the "first" and "second" eigenvalues
         that are needed to calculate the principal curvatures.
         If order ==3 and cm==1.5 then the reflect padding is used in the
         central difference computation. If order==3,cm==1, then zero padding.
         In all other cases for order==3, the same padding as in the
         convolution in the separate cm modes is used.
        -'ocg' specifies wether a strict or relaxed occupancy grid is used in
         the method itself. Use "str" for strict occgrid, "rlx" for relaxed.
        -'taulow' is the threshold for the values in the relaxed occgrid. If a 
         value is >taulow it is used in the relaxed occgrid and ignored 
         otherwise. This exists to limit the error that can happen due to the
         usage of a relaxed occupancy grid.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Uses a ball kernel for the neighborhood intersection.
    Returns gaussian curvature, both principal directions and the
    surface normal each in their own matrices where each voxel corresponds to
    the value.
    """
    if debug==1:
        starttime = time.time()    
        
    kappa1, kappa2, pd1, pd2, sn = cepca_pointcloud(inp, rho, kr=kr, 
                                                    order=order, cm=cm, 
                                                    ocg=ocg, taulow=taulow)
    
    #calculate gaussian curvature
    gc = kappa1 * kappa2
    
        
    if debug==1:
                print("Success!.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
        
    return gc, pd1, pd2, sn

def cegausspca_ocg(inp, kr=3, order=2, cm=1, debug=0):
    """
    Calculates the gaussian curvature of strict occupancy grid using pca and 
    integral invariants. Returns faussian curvature, both principal directions 
    and the surface normal.
    
    Input:
        -'inp' is a strict occupancy grid
        -'kr' is the kernel radius.
        -'cm' stands for convolution mode.If ==1, the fft with zero-padding is
         used. If cm<1, then the discrete convolution with a certain kind of
         padding is used. If cm==0, zero-padding, if cm==0.25, 'reflect', 
         if cm==0.50, 'nearest', if cm==0.75, 'mirror', if cm==0.95, 'wrap'.
        -'order' is the parameter that specifies the order of the eigenvalues.
         If order==0, then the order is not changed at all. 
         If order==1, then the eigenvalues and eigenvectors are ordered 
         according to the values of the eigenvalues. I.e. the biggest 
         eigenvalue is first, 2nd biggest is 2nd, etc etc.
         If order==2, then we use the error approximations of Pottmann07 to
         estimate which eigenvalues are the "first" and "second" eigenvalues
         that are needed to calculate the principal curvatures.
         If order ==3 and cm==1.5 then the reflect padding is used in the
         central difference computation. If order==3,cm==1, then zero padding.
         In all other cases for order==3, the same padding as in the
         convolution in the separate cm modes is used.
        -the debug parameter is just there to enable easier debugging, e.g.
         by printing certain statements at a time.
    
    Uses a ball kernel for the neighborhood intersection.
    Returns gaussian curvature, both principal directions and the
    surface normal each in their own matrices where each voxel corresponds to
    the value.
    """
    if debug==1:
        starttime = time.time()    
        
    kappa1, kappa2, pd1, pd2, sn = cepca_ocg(inp, kr=kr, order=order,
                                                 cm=cm)
    
    #calculate gaussian curvature
    gc = kappa1 * kappa2
    
        
    if debug==1:
                print("Success!.\n"
                      + "Current Runtime: " + str(time.time() - starttime))
        
    return gc, pd1, pd2, sn
  
#docs for the next algs are missing
def simpleboundaryboxcurvest_pointcloud(inp, rho, kr=3, order=2, cm=1, 
                                        ocg="rlx", taulow=0, splitvar="simp", 
                                        lex=0, rg=1, mask=0):
    #get dims of boundary boxes
    bbdims = split.computebbsize(kr=kr)
    
    #make list of boundary boxes
    if splitvar=="simp":
        out = split.simplebs_pointcloud(inp=inp, kr=kr, rg=rg, ocg=ocg, 
                                        taulow=taulow)
    else:
        out = split.spbs_pointcloud(inp=inp, lex=lex, kr=kr, rg=rg, ocg=ocg, 
                                    taulow=taulow)
    
    #get np array of all centers
    bbcntrslist = [item[1] for item in out]
    bbcntrs = np.zeros([len(bbcntrslist),3],int)
    
    for i in range(0, len(bbcntrslist)):
        bbcntrs[i] = bbcntrslist[i]
    
    #curvature estimation part
    output = []
    #go through list of bbs and estimate curvature for each one, however
    #just estimating curvature on each box is not enough as this naive
    #approach would lead to massive artifacts between boundary boxes. We
    #have to access the neighboring boundary boxes and make a new ocg that
    #is larger than the box
    for bb in out:
        #make new bigger occupancy grid that fully fits neighboring boxes
        ocg = np.zeros([3*bbdims,3*bbdims,3*bbdims],float)
        ocgc = (3*bbdims)//2
        
        #find all possible neighboring centers (not 100% sure this hits all
        #required centers)
        cntrslow = bbcntrs >= [bb[1][0]-bbdims,bb[1][1]-bbdims,bb[1][2]-bbdims]
        cntrstop = bbcntrs <= [bb[1][0]+bbdims,bb[1][1]+bbdims,bb[1][2]+bbdims]
        cntrs = cntrslow * cntrstop
        cntrs = cntrs[:,0] * cntrs[:,1] * cntrs[:,2]
        cntrsidcs = np.where(cntrs==1)
        cntrs = bbcntrs[cntrsidcs]
        
        #fill big ocg with values from the boxes; go through all neighboring
        #center coordinates
        for i in range(0,np.shape(cntrsidcs)[1]):
            #measure distance between the current bb center and the center of
            #the initial boundary box (the latter is also the center of the
            #new and bigger occupancy grid)
            currcntr = cntrs[i]
            dist = [bb[1][0]-currcntr[0],bb[1][1]-currcntr[1],
                    bb[1][2]-currcntr[2]]
            #then fill the big ocg with all values of the relevant bbs
            ocg[ocgc-dist[0]-bbdims//2:\
                ocgc-dist[0]+bbdims//2,\
                ocgc-dist[1]-bbdims//2:\
                ocgc-dist[1]+bbdims//2,
                ocgc-dist[0]-bbdims//2:
                ocgc-dist[0]+bbdims//2] = out[cntrsidcs[0][i]][0]
        
        #actual curvature estimation
        #cebb = cepca_ocg(bb[0], kr=kr, order=order, cm=cm)
        ceocg = cepca_ocg(ocg, kr=kr, order=order, cm=cm)
        
        #now put together the original boundary box......and output it
        output.append([[ceocg[0][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2],
                        ceocg[1][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2],
                        ceocg[2][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,:],
                        ceocg[3][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,:],
                        ceocg[4][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,:]],bb[1]])
    
    if mask==0:
        return output
    elif mask==1:
        return output, out

def simpleboundaryboxcurvest_ocg(inp, kr=3, order=2, cm=1, ocg="rlx", taulow=0,
                                 splitvar="simp", mask=0):
    #get dims of boundary boxes
    bbdims = split.computebbsize(kr=kr)
    
    #make list of boundary boxes
    if splitvar=="simp":
        out = split.simplebs_ocg(inp=inp, kr=kr, ocg=ocg, taulow=taulow)
    else:
        out = split.spbs_ocg(inp=inp, kr=kr,  ocg=ocg, taulow=taulow)
    
    #get np array of all centers
    bbcntrslist = [item[1] for item in out]
    bbcntrs = np.zeros([len(bbcntrslist),3],int)
    
    for i in range(0, len(bbcntrslist)):
        bbcntrs[i] = bbcntrslist[i]
    
    output = []
    #go through list of bbs and estimate curvature for each one, however
    #just estimating curvature on each box is not enough as this naive
    #approach would lead to massive artifacts between boundary boxes. We
    #have to access the neighboring boundary boxes and make a new ocg that
    #is larger than the box
    for bb in out:
        #make new bigger occupancy grid that fully fits neighboring boxes
        ocg = np.zeros([3*bbdims,3*bbdims,3*bbdims],float)
        ocgc = (3*bbdims)//2
        
        #find all possible neighboring centers (not 100% sure this hits all
        #required centers)
        cntrslow = bbcntrs >= [bb[1][0]-bbdims,bb[1][1]-bbdims,bb[1][2]-bbdims]
        cntrstop = bbcntrs <= [bb[1][0]+bbdims,bb[1][1]+bbdims,bb[1][2]+bbdims]
        cntrs = cntrslow * cntrstop
        cntrs = cntrs[:,0] * cntrs[:,1] * cntrs[:,2]
        cntrsidcs = np.where(cntrs==1)
        cntrs = bbcntrs[cntrsidcs]
        
        #fill big ocg with values from the boxes; go through all neighboring
        #center coordinates
        for i in range(0,np.shape(cntrsidcs)[1]):
            #measure distance between the current bb center and the center of
            #the initial boundary box (the latter is also the center of the
            #new and bigger occupancy grid)
            currcntr = cntrs[i]
            dist = [bb[1][0]-currcntr[0],bb[1][1]-currcntr[1],
                    bb[1][2]-currcntr[2]]
            #then fill the big ocg with all values of the relevant bbs
            ocg[ocgc-dist[0]-bbdims//2:\
                ocgc-dist[0]+bbdims//2,\
                ocgc-dist[1]-bbdims//2:\
                ocgc-dist[1]+bbdims//2,
                ocgc-dist[0]-bbdims//2:
                ocgc-dist[0]+bbdims//2] = out[cntrsidcs[0][i]][0]
        
        #actual curvature estimation
        #cebb = cepca_ocg(bb[0], kr=kr, order=order, cm=cm)
        ceocg = cepca_ocg(ocg, kr=kr, order=order, cm=cm)
        
        #now put together the original boundary box......and output it
        output.append([[ceocg[0][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2],
                        ceocg[1][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2],
                        ceocg[2][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,:],
                        ceocg[3][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,:],
                        ceocg[4][ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,
                                 ocgc-bbdims//2:ocgc+bbdims//2,:]],bb[1]])
        
    if mask==0:
        return output
    elif mask==1:
        return output, out


def cemean_bb_principalcurv(inp,kr):
    #split input 'inp' into principal curvatures of the boundary boxes
    kap1 = [item[0][0] for item in inp]
    kap2 = [item[0][1] for item in inp]
    
    out = []
    for i in range(0, len(kap1)):
        out.append([(kap1[i]+kap2[i])/2,inp[i][1]])
    
    return out

def cegauss_bb_principalcurv(inp,kr):
    #split input 'inp' into principal curvatures of the boundary boxes
    kap1 = [item[0][0] for item in inp]
    kap2 = [item[0][1] for item in inp]
    
    out = []
    for i in range(0, len(kap1)):
        out.append([kap1[i]*kap2[i],inp[i][1]])
    
    return out
    
    
    
