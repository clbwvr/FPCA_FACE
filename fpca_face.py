"""
Functional principal component analysis with fast covariance estimation

A fast implementation of the sandwich smoother (Xiao et al., 2013)
for covariance matrix smoothing.

Implemented by Caleb Weaver (cjweave2@ncsu.edu)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from math import factorial
import numpy.linalg as la
from scipy.optimize import minimize
from scipy.interpolate import splev

class FPCA_FACE:
    """
    FACE Algorithm for FPCA.
    
    Inputs
    - Y: I by J data matrix, functions on rowses.
    - argvals: grid of length J
    - knots: number of knots
    - p: degree of b-splines
    - m: order of difference penalty
    - npc: number of PCs
    - center: flag for setting columns means of Y to zero
    
    Outputs
    - Yhat: Smoothed Y
    - scores - Matrix of Scores
    - mu - Mean function
    - efunctions: Eigenfunctions
    - evalues: Eigenvalues
    - fve: Function variance explained by each PC
    """
    def __init__(self, Y, argvals = None, knots = 35, p = 3, m=2, npc = 10, center = True):
        self.Y = Y
        self.argvals = argvals
        self.knots = knots
        self.p = p
        self.m = m
        self.Yhat = None
        self.mu = None
        self.efunctions = None
        self.evalues = None
        self.scores = None
        self.npc = npc
        self.center = center
        self.fve = None
        
    def fit(self):
        Y = pd.DataFrame(self.Y)
        data_dim = Y.shape
        I = data_dim[0] 
        J = data_dim[1]
        
        if(self.argvals is None):
            self.argvals = [(j+1)/J-1/2/J for j in range(J)]
     
        # Get basis for B-spline
        knots =  np.linspace(-self.p,self.knots+self.p,num = self.knots + 1 + 2 * self.p)/self.knots
        knots = knots*(np.max(self.argvals) - np.min(self.argvals)) + np.min(self.argvals)
        K = len(knots)-2*self.p-1
        mk = len(knots) - 4
        v = np.zeros((mk, len(self.argvals)))
        d = np.eye(mk, len(knots))
        for i in range(mk):
            v[i] = splev(self.argvals, (knots, d[i], 4-1), der = 0)
        B = v.T
        
        # Center data
        meanY = [0 for j in range(J)]
        if self.center:
            meanY = np.mean(Y,axis=0)
            lmfit = LinearRegression().fit(B, meanY)
            meanY = lmfit.predict(B)
            Y = Y - meanY
            
        def difference_penalty(m,p,K):
            c = [0 for _ in range(m+1)]
            for i in range(m+1):
              c[i] = (-1)**(i+1)*factorial(m)/(factorial(i)*factorial(m-i))
            M = np.zeros([K+p-m,K+p])
            for i in range(K+p-m):
                M[i,i:(i+m+1)] = c
            return(M)
        
        P = difference_penalty(self.m,self.p,K)
        P1 = P
        P2 = P.T
        P = P2 @ P1
    
        def MM(A,s,option):
            if option==1:
                c = [1 for _ in range(A.shape[0])]
                c = np.reshape(c,[len(c),1])
                s = np.reshape(s,[1,len(s)])
                return(A * (c @ s))
            if option==2:
                c = [1 for _ in range(A.shape[1])]
                c = np.reshape(c,[1,A.shape[1]])
                s = np.reshape(s,[A.shape[0],1])
                return((A * (s @ c)))
        
        weight = np.array([1 for _ in range(len(self.argvals))])
        B1 = MM(B.T,weight,1)
        Sig = B1 @ B
        esig = la.eig(Sig)
        V = esig[1]
        E = esig[0]
        if np.min(E) <= 1e-6:
            E = E + 1e-6
                       
        sigi_sqrt = MM(V,1/np.sqrt(E),1) @ V.T
        
        print(hd(sigi_sqrt))
        
        tupu = sigi_sqrt @ (P @ sigi_sqrt)
        esig = la.eig(tupu)
        U = esig[1]
        s = esig[0]
        s[(K+self.p-self.m):(K+self.p)]=0
        A = B @ sigi_sqrt @ U
        Bt = B.T
        A0 = sigi_sqrt @ U
        
        Ytilde = (A0.T).dot(Bt.dot(Y.T))
        C_diag = np.sum(np.power(Ytilde,2),axis=1)
       
        # Select smoothing parameters
        Y_square = np.sum(np.sum(np.power(Y,2)))
        Ytilde_square = np.sum(np.sum(np.power(Ytilde,2)))
        
        def face_gcv(x):
          lam = np.exp(x)
          lam_s = ((lam*s)**2)/((1 + lam*s)**2)
          gcv = np.sum(C_diag*lam_s) - Ytilde_square + Y_square
          trace = np.sum(1/(1+lam*s))
          gcv = gcv/(1-trace/J)**2
          return(gcv)
    
        
        res = minimize(face_gcv,0)
        lam = np.exp(res.x)
        print(lam)
        
        YS = MM(Ytilde,1/(1+lam*s),option=2)
    
        # Eigendecompositon of Smoothed Data
        temp = (YS) @ (YS.T/I)
        Eigen = la.eig(temp)
        sigma = Eigen[0]/J
        A = Eigen[1]
        
        # Functional Variance Explained
        d = sigma[:self.npc]
        fve = d/np.sum(d)
        
        Ypred = Y
        Ytilde = ((A0.T) @ (Bt)) @ (Ypred.T.values)
        sigmahat2 = np.max(np.mean(np.mean(Y**2)) - np.sum(sigma),0)
        Xi = (Ytilde.T) @ (A[:,:self.npc]/np.sqrt(J))
        Xi = MM(Xi,sigma[:self.npc]/(sigma[:self.npc] + sigmahat2/J),option=1)
        
        eigenvectors = B @ (A0 @ A[:,:(self.npc)])
        eigenvalues = sigma[:self.npc]
        Yhat = (A[:,:self.npc]).T @ Ytilde
        Yhat = B @ A0 @ A[:,:self.npc] @ np.diag(eigenvalues/(eigenvalues+sigmahat2/J)) @ Yhat
        Yhat = (Yhat.T + meanY)
        
        scores = np.sqrt(J)*Xi[:,:self.npc]
        mu = meanY
        efunctions = np.real(eigenvectors[:,:self.npc])
        evalues = np.real(J*eigenvalues[:self.npc])
        
        self.Yhat = Yhat
        self.scores = scores
        self.mu = mu
        self.efunctions = efunctions
        self.evalues = evalues
        self.fve = fve
        
        return(self)
