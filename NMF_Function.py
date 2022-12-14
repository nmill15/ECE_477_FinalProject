# Functions for Final Project
# ECE 477
# Author: Noah Miller

import numpy as np

def myNMF(V, r, nIter, initW=None, initH=None, bUpdateW=1, bUpdateH=1):
    #  Implementation of the multiplicative update rule for NMF with K-L
    #  divergence. W should always be normalized so that each column of W sums
    #  to 1. Scale H accordingly. This normalization step should be performed in
    #  each iteration, and in initialization as well. This is to deal with the
    #  non-uniqueness of NMF solutions.
    
    #  Input
    #    - V         : a non-negative matrix (m*n) to be decomposed
    #    - r         : #columns of W (i.e., #rows of H)
    #    - nIter     : #iterations
    #    - initW     : initial value of W (m*r) (default: a random non-negative matrix)
    #    - initH     : initial value of H (r*n) (default: a random non-negative matrix)
    #    - bUpdateW  : update W (bUpdateW==1) or not (bUpdateW==0) (default: 1)
    #    - bUpdateH  : update H (bUpdateH==1) or not (bUpdateH==0) (default: 1)
    
    #  Output
    #    - W         : learned W
    #    - H         : learned H
    #    - KL        : KL divergence after each iteration (a row vector with nIter+1
    #                elements). KL(1) is the KL divergence using the initial W
    #                and H.


    [m,n] = V.shape # Finding the dimensions for the deconstruction

    KL = np.zeros(nIter+1) # KL Divergence Array

    # Checking for initial approximations 
    if initW is None:
        W = np.random.rand(m,r)
    else: 
        W = initW
    if initH is None:
        H = np.random.rand(r,n)
    else: 
        H = initH

    # Looping through the number of iterations
    for j in range(nIter):

        WH = W @ H # Matrix Multiplication of both arrays

        KL[j+1] = np.sum(V * np.log(V/WH) - V + WH) # Calculating the KL Divergence

        # Multiplicative Update 
        if(bUpdateW == 1):
            for i in range(m):
                    for a in range(r):
                        W[i,a] *= np.sum(H[a]*V[i]/(WH[i])/np.sum(H[a]))
                        
        if(bUpdateH == 1):
            for a in range(r):
                for mu in range(n):
                    H[a,mu] *= np.sum(W.T[a]*V.T[mu]/WH.T[mu])/np.sum(W.T[a])

        # Normalizng the approximations
        norm = np.sum(W,axis=0)
        W *= 1.0 / (norm)
        

    return [W,H,KL]