#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decompose a matrix X using reduced basis decomposition. X = YT. Y basis vectors
are constructed using mod_gramm_schmidt until an maximum basis number threshold
(dmax) or an error requirement (Er) are satisfied.
"""

__author__ = "Anthony Abercrombie"

import numpy as np
import matplotlib.image as mpimg
from error_statistics import calculate_error_stats

def matrix_to_attributes(X):
    '''
    Read in a matrix X and establish core attributes for RBD decomposition.
    '''
    #Ensure the input matrix X will be an array of floats
    if type(X) == list:
        X = np.array(X).astype(np.float128)

    elif type(X) == np.ndarray:
        X = X.astype(np.float128)
    else:
        raise TypeError('Input must be a list or an np.ndarray')
    # d is used to track the orthogonal vector that is under construction.
    # It increases in steps of one.
    d = 0
    # e_cur is a measure of maximum error in regards to the column vectors of X
    # and the column vectors of the decomposition.
    e_cur = np.inf
    #i is used to index the input matrix X. i is determined so that X[:,i]
    # corresponds with e_cur
    i = np.random.randint(0, X.shape[1])
    # Y is an orthogonal matrix of basis vectors that is grown sequentially by
    # the Gramm-Schmidt process. The dot product of Y and T produces a
    # compression of X.
    Y = np.zeros(shape=(X.shape[0], 1)).astype(np.float128)
    T = np.zeros(shape=(1,X.shape[1])).astype(np.float128)
    # used_i holds the indexes of X used to construct Y and T in the order they
    # were selected. Each element in used_i should be unique.
    used_i = []
    #complete is used to halt the algorithm and is subject to change if the
    # error requirement Er is met.
    complete = False

    attributes = (X,Y,T,d,e_cur, i,used_i, complete)
    return attributes

def mod_gramm_schmidt(X,Y,T,d,e_cur,Er,i,used_i,complete):
    """
    Orthogonally project an X column vector into the orthonormal Y basis.
    """
    # Pick a vector from the input matrix X
    Xvector = X[:, i]
    # If this is the first vector to be projected into the orthonormal basis Y..
    if d == 0:
        # Normalize the vector by dividing the vector by its magnitude
        Xvector_norm = np.linalg.norm(Xvector)
        first_Ybasis = Xvector/Xvector_norm
        # Change the 1st column of Y from a zero vector to the normalized vector
        Y[:, d] = first_Ybasis.copy()
        # Change the 1st row of T from a zero vector to the dot product of the
        # normalized vector's transpose and X. This ensures that an image of X
        # can be reconstructed by the dot product of Y and T.
        T[d, :] = np.dot(Y[:,d].T, X)
        # append the index of the chosen vector to a list for book-keeping purposes.
        used_i.append(i)
        # Return an updated state of all the RBD parameters, leading to the
        # find_error process of the algorithm.
        update_state = (X,Y,T,d,e_cur,Er,i,used_i,complete)
        return update_state

    # Otherwise, Gramm-Schmidt is used to orthogonally project a chosen vector
    # into the Y space.
    else:
        #u_d is the dth vector to be chosen by the find_error module.
        u_d = Xvector.copy()
        for j in range(d):
            #Subtract the projection of u_d onto the jth vector in Y from u_d
            proj = np.dot(u_d, Y[:,j]) * Y[:,j]
            u_d -= proj
        # If the magnitude of u_d, which represents its error correcting power,
        # falls below the error requirement, complete the RBD decomposition process.
        if np.linalg.norm(u_d) <= Er:
            Y = Y[:, d-1]
            T = T[:, d-1]
            update_complete = True
            #return parameters with complete == True signaling the end of RBD.
            decomposition = (X,Y,T,d,e_cur,Er,i,used_i,update_complete)
            return decomposition
        #If the error requirement has not been met, normalize u_d and add it to
        # Y as a column vector. Add the dot product of u_d.transpose and X to T
        else:
            basis_vector = np.divide(u_d, np.linalg.norm(u_d))
            Y = np.column_stack((Y, basis_vector))
            T = np.row_stack((T, np.dot(Y[:,d].T, X)))
            used_i.append(i)
            #Return updated parameters, leading to the find_error process of the algorithm.
            update_state = (X,Y,T,d,e_cur,Er,i,used_i,complete)
            return update_state

def find_error(X,Y,T,d,e_cur,Er,i,used_i,complete):
    """
    Evaluate the error discrepancy of each X column vector and the current
    decomposition's column vectors and select the optimal X vector for the next
    round of mod_gramm_schmidt.
    """
    # Calculate the error discrepancy of between each X column vector and the
    # decomposition's corresponding column vector.
    error_list = [np.linalg.norm(X[:,j] - np.dot(Y, T[:,j])) for j in range(X.shape[1])]
    # current error is set to the highest error from the comparisons.
    new_e_cur = np.max(error_list)
    # The next column-index of X to project into the Y space is selected as the
    # column vector with the highest error discrepancy.
    next_i = np.argmax(error_list)
    # Return updated parameters, leading to the check_error function
    updated_attributes = X,Y,T,d,new_e_cur,Er,next_i,used_i,complete
    return updated_attributes

def check_error(X,Y,T,d,e_cur,Er,i,used_i,complete):
    """
    Evaluate whether the decomposition is complete and prepare for the next round
    of mod_gramm_schmidt
    """

    # If the error discrepancy of the next X column vector meets the error
    # requirement, finalize the RBD process.
    if e_cur <= Er:
        Y = Y[:, :d]
        T = T[:d, :]
        # Complete == True signals the end of the process.
        update_complete = True
        update_attributes = X,Y,T,d,e_cur,Er,i,used_i,update_complete
    # Otherwise, there is a need to continue adding basis vectors to Y. Increasing d is the last step before repeating the gramm-schmidt process.
    else:
        grow_d = d + 1
        update_attributes = X,Y,T,grow_d,e_cur,Er,i,used_i,complete
    # Return the parameters to the main function: rbd
    return update_attributes

def rbd(X, dmax, Er=0.01):
    """
    Decompose a matrix X using reduced basis decomposition. X = YT. Y basis
    vectors are constructed using mod_gramm_schmidt until an maximum basis
    number threshold (dmax) or an error requirement (Er) are satisfied.
    """
    # Call matrix_to_attributes from the setup.py module to instantiate the parameters for the RBD
    X,Y,T,d,e_cur, i,used_i, complete = matrix_to_attributes(X)
    # Set the current_state to the initial parameters; subject to iterative change.
    current_state = [X,Y,T,d,e_cur,Er,i,used_i,complete]
    # While the requirements for the RBD have not been met...
    while d <= dmax and e_cur > Er and complete == False:
        # Orthogonally project the X[:,i] vector into Y, calling the mod_gramm_schmidt.py module.
        orthonormalization = mod_gramm_schmidt(*current_state)
        # mod_gramm_schmidt has a clause that signals complete to equal True.
        complete = orthonormalization[-1]
        # Break the while loop if mod_gramm_schmidt tells us the decomposition is complete.
        if complete == True:
            current_state = orthonormalization
            break
        else:
            # Prepare for the next round of orthonormalization by calling the
            # find_error.py and check_error.py modules.
            current_state  = check_error(*find_error(*orthonormalization))
            # Update the active parameters, some of which govern the while loop.
            # check_error() has a clause that signals completion.
            X,Y,T,d,e_cur,Er,i,used_i,complete = current_state
    # Final output of the function is a list of all the parameters.
    return current_state

if __name__ == '__main__':
    alist = np.array([[1,1,0,1,0],[1,0,1,1,0],[0,1,1,1,0]])
    test_rbd = rbd(alist, dmax=4)

    def display_state(attributes):
        names = ['X', 'Y', 'T','d' ,'e_cur', 'Er', 'i', 'used_i', 'complete']
        for i in range(len(names)):
            print("\n\n", names[i], " :\n", attributes[i])


    def image_test():
        image = mpimg.imread('images/DrawingHands.jpg')
        #tenbyten = np.array(image)[:,:,0]
        tenbyten = image
        tenbyten_test = rbd(tenbyten, dmax=30, Er = 0.05)
        X,Y,T,d,e_cur,Er,i,used_i,complete = tenbyten_test
        decomp = np.dot(Y,T)
        print("DECOMP: \n", decomp)
        print("INPUT: \n", X)

        display_state(tenbyten_test)
        MSE, NMSE, RMSD, NRMSD, PSNR = calculate_error_stats(X, decomp)

        print("MSE: ", MSE)
        print("NMSE: ", NMSE)
        print("RMSD: ", RMSD)
        print("NRMSD: ", NRMSD)
        print("PSNR: ", PSNR)
