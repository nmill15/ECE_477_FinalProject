# Functions for HW #4 
# ECE 477
# Author: Noah Miller

import numpy as np
import scipy
from IPython.display import Audio

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

def myViterbi(transMat, loglikeMat, initProb):
    #  Implementation of the Viterbi algorithm to find the path of states that has the
    #  highest posterior probability in a finite-state hidden Markov model
    #  (HMM).
    
    #  Input
    #    - transMat      : the transition probability matrix, P(S_n | S_n-1).
    #                        Rows correspond to the starting states; columns
    #                        corresond to the ending states. Each row sums to 1.
    #                        (size: nState * nState)
    #    - loglikeMat    : the log-likelihood matrix of each state to explain
    #                        each observation, log( P(O_n | S_n) ) Rows
    #                        correspond to states; columns correspond to
    #                        observations. (size: nState * nObserv)
    #    - initProb      : the initial probability of all states, P(S_0). A
    #                        column vector with nState elements.
    
    #  Output
    #    - path          : the best path of states (S_1, ..., S_N) that gives
    #                        the maximal posterior probability, P(S_1, ..., S_N
    #                        | O_1, ... O_N). A column vector with nObserv elements.

    # Initializing matrices
    numSta = transMat.shape[0]
    numObs = loglikeMat.shape[1]
    V = np.zeros((numSta, numObs))
    past = np.zeros((numSta, numObs))
    path = np.zeros(numObs)

    V[:,0] = np.log(initProb[:,0]) + loglikeMat[:,0] # Initial value 

    for i in range(1, numObs):
        for j in range(numSta):
            delt = np.log(transMat[:,j]) + V[:,i-1]
            V[j,i] = np.max(delt) + loglikeMat[j,i]
            past[j,i] = np.argmax(delt)

    # Backtracing
    path[-1] = np.argmax(past[:,-1])

    for i in range(numObs-1,-1,-1):
        path[i - 1] = past[int(path[i]), i]
    
    return path



