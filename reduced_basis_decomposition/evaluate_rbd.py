#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate performance of reduced basis decomposition on a directory of images.
"""

__author__ = "Anthony Abercrombie"

import numpy as np
import math
import pandas as pd
import matplotlib.image as mpimg
from reduced_basis_decomposition import *
from timeit import default_timer
import os
import glob

def decomposition_error(X, decomp):
    """
    Function for capturing error statistics of a particular decomposition.
    Calculates Mean Square Error (MSE), Normalized Mean Square Error (NMSE),
    Peak Signal to Noise Ratio (PSNR), Root Mean Standard Deviation (RMSD),
    and Normalized Root Mean Standard Deviation (NRMSD).
    """
    # Collect square errors of each corresponding cells in X and the decomposition
    square_errors = []
    square_xs = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # sq_er metric is all that is needed for MSE
            sq_er = (X[i,j] - decomp[i,j])**2
            square_errors.append(sq_er)
            # square_x norm is the distance from a cell in the decomposition and
            # the point of origin. Necessary for the NMSE metric.
            square_xs.append(decomp[i,j]**2)
    # Mean Square Error as the sum of squares divided by the size of the matrix.
    MSE = np.sum(square_errors)/X.size

    # Normalized Mean Square Error
    NMSE_denominator = np.sum(square_xs)/X.size
    NMSE= MSE/NMSE_denominator

    # Peak Signal to Noise Ratio
    PSNR = 20 * math.log((255/np.sqrt(MSE)), 10)

    # Root Mean Standard Deviation
    RMSD = np.sqrt(MSE)

    # Normalized Root Mean Standard Deviation.
    NRMSD = RMSD/np.mean(square_errors)

    return MSE, NMSE, RMSD, NRMSD, PSNR

def rbd_steps_error(matrix, return_df=False):
    """
    Calculates error and runtime statistics for RBD at each instance of dmax.
    Option to return output as a Pandas DataFrame.
    """
    # Setup list repositories.
    number_of_basis = []
    runtime = []
    e_cur_data = []
    MSE_data = []
    NMSE_data = []
    RMSD_data = []
    NRMSD_data = []
    PSNR_data = []
    for j in range(matrix.shape[1]):
        # Capture the runtime of RBD
        start = default_timer()
        X,Y,T,d,e_cur,Er,i,used_i,complete = rbd(matrix, dmax=j, Er= -np.inf)
        end = default_timer()
        # Gather error metrics using the error_statistics.py module
        MSE, NMSE, RMSD, NRMSD, PSNR = decomposition_error(X, np.dot(Y,T))
        # Append values to list repositories.
        number_of_basis.append(d)
        runtime.append(end - start)
        e_cur_data.append(e_cur)
        MSE_data.append(MSE)
        NMSE_data.append(NMSE)
        RMSD_data.append(RMSD)
        NRMSD_data.append(NRMSD)
        PSNR_data.append(PSNR)
    if return_df is True:
        # Setup Pandas DataFrame and return.
        stats_dataframe = pd.DataFrame(
            {'basis_count': number_of_basis,
            'runtime': runtime,
            'e_cur': e_cur_data,
            'MSE': MSE_data,
            'NMSE': NMSE_data,
            'RMSD': RMSD_data,
            'NRMSD': NRMSD_data,
            'PSNR': PSNR_data})
        return stats_dataframe
    else:
        # Return each series as lists
        return number_of_basis, runtime, e_cur, MSE_data, NMSE_data, RMSD_data, NRMSD_data, PSNR_data

def evaluate_rbd_on_directory(directory, return_df=False):
    """
    Crawls through the image directory and gathers RBD statistics on each .pgm
    image in the directory. Option to return a flattened Pandas DataFrame. This
    function takes a very long time to run.
    """
    # Setup repositories
    huge_number_of_basis = []
    huge_runtime = []
    huge_e_cur_data = []
    huge_MSE_data = []
    huge_NMSE_data = []
    huge_RMSD_data = []
    huge_NRMSD_data = []
    huge_PSNR_data = []
    for filename in glob.glob(os.path.join(directory,'*.pgm')):
        image = mpimg.imread(filename)
        # Use the rbd_steps_error function from the gather_basis_statistics.py
        # module to collect data.
        number_of_basis, runtime, e_cur, MSE_data, NMSE_data, RMSD_data,NRMSD_data, PSNR_data = rbd_steps_error(image, return_df=False)
        # Append observations to data repositories
        huge_number_of_basis.append(number_of_basis)
        huge_runtime.append(runtime)
        huge_e_cur_data.append(e_cur)
        huge_MSE_data.append(MSE_data)
        huge_NMSE_data.append(NMSE_data)
        huge_RMSD_data.append(RMSD_data)
        huge_NRMSD_data.append(NRMSD_data)
        huge_PSNR_data.append(PSNR_data)
    # Flatten the repositories and alter their datatypes
    huge_number_of_basis = np.array(huge_number_of_basis).flatten()
    huge_runtime = np.array(huge_runtime).flatten().astype(float)
    huge_e_cur_data = np.array(huge_e_cur_data).flatten().astype(float)
    huge_MSE_data = np.array(huge_MSE_data).flatten().astype(float)
    huge_NMSE_data = np.array(huge_NMSE_data).flatten().astype(float)
    huge_RMSD_data = np.array(huge_RMSD_data).flatten().astype(float)
    huge_NRMSD_data = np.array(huge_NRMSD_data).flatten().astype(float)
    huge_PSNR_data = np.array(huge_PSNR_data).flatten().astype(float)
    if return_df is True:
      # Setup the Pandas DataFrame.
        stats_dataframe = pd.DataFrame(
            {'basis_count': huge_number_of_basis,
            'runtime': huge_runtime,
            'e_cur': huge_e_cur_data,
            'MSE': huge_MSE_data,
            'NMSE': huge_NMSE_data,
            'RMSD': huge_RMSD_data,
            'NRMSD': huge_NRMSD_data,
            'PSNR': huge_PSNR_data})
        return stats_dataframe
    else:
        # Return each series as lists
        return huge_number_of_basis, huge_runtime, huge_e_cur_data, huge_MSE_data, huge_NMSE_data, huge_RMSD_data, huge_NRMSD_data, huge_PSNR_data



if __name__ == '__main__':
      image = mpimg.imread('images/1a000.pgm')
      matrix = image
      #matrix = np.array(image)[:,:,0]
      stats_dataframe = rbd_steps_error(matrix)
