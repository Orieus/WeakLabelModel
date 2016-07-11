#!/usr/bin/env python
# -*- coding: utf-8 -*-

# External modules
import numpy as np
from numpy import binary_repr
import sklearn.datasets as skd           # Needs version 0.14 or higher
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt

def generateWeak(y,M,c):
    """
   Generate the set of weak labels z for n examples, given the ground truth 
   labels y for n examples, a mixing matrix M, and a number of classes c.
    """

    z = np.zeros(y.shape)                # Weak labels for all labels y (int)
    d = 2**(c)                           # Number of weak labels
    dec_labels = np.arange(d)            # Possible weak labels (int) 

    for index,i in enumerate(y):

    	 z[index] = np.random.choice(dec_labels, 1, p=M[:,i]) 

    return z

def generateVirtual(z,M,c,method ='equal'): 
    """
   Generate the set of virtual labels v for n examples, given the weak labels 
   for n examples in decimal format, a mixing matrix M, the number of classes c, 
   and a method.
    """
    
    z_bin = np.zeros((z.size,c))         # weak labels (binary)
    v = np.zeros((z.size,c))             # virtual labels 

    for index,i in enumerate(z):         # From dec to bin

        z_bin[index,:] = [int(x) for x in bin(int(i))[2:].zfill(c)]
    
    if method == 'equal':                # weak and virtual are the same

    	v = z_bin

    if method == 'independent_noisy':    # quasi-independent labels       
        
        for index,i in enumerate(z_bin):

            aux = z_bin[index,:]
            weak_pos = np.sum(aux)
            
            if not weak_pos == c:

                weak_zero = (1-weak_pos)/(c-weak_pos)
                aux[aux == 0] = weak_zero
                z_bin[index,:] = aux

            else:

            	z_bin[index,:] = np.array([None] * c)   

    return v   

def generateM(c,alpha=0.5,beta=0.5,gamma=0.5, method='supervised'):
    """
   Generate a mixing matrix M, given the number of classes c.
    """

    if method == 'supervised':

        M = np.array([[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        	[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])    	

    if method == 'single_noisy':

        M = np.array([[0.0, 0.0, 0.0], [1-alpha, beta/2, gamma/2],
    	    [alpha/2, 1-beta, gamma/2],[alpha/2, beta/2, 1-gamma],
    	    [0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])

    if method == 'independent_noisy':

        M = np.array([[0.0, 0.0, 0.0], [1-alpha-alpha**2, 0, 0],
    	    [0, 1-beta-beta**2, 0],[0, 0, 1-gamma-gamma**2],
    	    [alpha/2, beta/2, 0.0],[alpha/2, 0.0, gamma/2],
    	    [0.0, beta/2, gamma/2],[alpha**2, beta**2, gamma**2]])        

    return M

# ##############################################################################
# ## MAIN ######################################################################
# ##############################################################################

############################
# ## Configurable parameters

# Parameters for sklearn synthetic data
ns = 100    # Sample size
nf = 2      # Data dimension
c = 3       # Number of classes 


#####################
# ## A title to start

print "======================="
print "    Weak labels"
print "======================="

###############################################################################
# ## PART I: Load data (samples and true labels)                             ##
###############################################################################

X, y = skd.make_classification(
    n_samples=ns, n_features=nf, n_informative=2, n_redundant=0,
    n_repeated=0, n_classes=c, n_clusters_per_class=1, weights=None,
    flip_y=0.0001, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
    shuffle=True, random_state=None)


M = generateM(c,alpha=0.5,beta=0.5,method='independent_noisy')
z = generateWeak(y,M,c)
v = generateVirtual(z,M,c,method='independent_noisy')


