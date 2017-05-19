#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_reduced_basis_decomposition
----------------------------------

Tests for `reduced_basis_decomposition` module.
"""


import sys
import unittest
from contextlib import contextmanager
from click.testing import CliRunner

from reduced_basis_decomposition import reduced_basis_decomposition
from reduced_basis_decomposition.reduced_basis_decomposition import *
from reduced_basis_decomposition import cli

import urllib2



class TestReduced_basis_decomposition(unittest.TestCase):

    def setUp(self):
        print('In setUp()')
        f = urllib2.urlopen("https://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png")
        self.test_image = mpimg.imread(f)

    def tearDown(self):
        print('In tearDown()')
        del self.test_image

    def test_matrix_to_attributes(self):
        '''
        Test harness for the matrix_to_attributes function
        '''
        dmax = 40
        X,Y,T,d,e_cur, i,used_i, complete = matrix_to_attributes(self.test_image)

        # Test datatypes and values
        assert type(X) == np.ndarray
        assert type(Y) == np.ndarray
        assert type(T) == np.ndarray
        assert type(d) == int
        assert type(e_cur) == float
        assert type(i) == int
        assert type(used_i) == list
        assert complete == False
        assert d < dmax

        # Test shapes
        assert Y.shape[0] == X.shape[0]
        assert Y.shape[1] == 1
        assert T.shape[0] == 1
        assert T.shape[1] == X.shape[1]
        assert len(used_i) == 0
        assert d < dmax

    def test_d0(self):
    	"""
        Test the first iteration of mod_gramm_schmidt where d == 0
        """
    	# Setup parameters
    	X,Y,T,d,e_cur, i,used_i, complete = matrix_to_attributes(self.test_image)
    	Er = 0.0000000001

    	# Apply gramm_schmidt
    	gramm_schmidt_results = mod_gramm_schmidt(X,Y,T,d,e_cur,Er,i,used_i, complete)

    	# Update testing parameters
    	X,Y,T,d,e_cur,Er,i,used_i,complete = gramm_schmidt_results

       # Test datatypes and values
    	assert type(X) == np.ndarray
    	assert type(Y) == np.ndarray
    	assert type(T) == np.ndarray
    	assert type(d) == int

    	assert type(Er) == float
    	assert type(i) == int
    	assert type(used_i) == list
    	assert complete == False

    	# Test that Y and T have non-zero basis
    	assert np.count_nonzero(Y) > 0
    	assert np.count_nonzero(T) > 0

    	# Test that Y and T have the right shapes.
    	assert Y.shape[1] == 1
    	assert Y.shape[0] == X.shape[0]
    	assert T.shape[1] == X.shape[1]
    	assert T.shape[0] == 1

    	# Assert that an index has been added to used_i"""
    	assert len(used_i) == 1

    def test_d1_not_done(self):
    	"""
        Test mod_gramm_schmidt where d!=0 and there is a need to continue
        building the basis.
        """
    	# Setup parameters
    	X,Y,T,d,e_cur, i,used_i, complete = matrix_to_attributes(self.test_image)
    	Er = 0.0000000001

    	# Complete 1 round of the RBD decomposition algorithm
    	d1_parameters = check_error(*find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete)))

    	# Apply gramm_schmidt at the beginning of stage d = 1
    	gramm_schmidt_results = mod_gramm_schmidt(*d1_parameters)

    	# Update testing parameters
    	X,Y,T,d,e_cur,Er, i,used_i, complete = gramm_schmidt_results

    	# Test datatypes and values
    	assert type(X) == np.ndarray
    	assert type(Y) == np.ndarray
    	assert type(T) == np.ndarray
    	assert type(d) == int
    	assert type(Er) == float
    	assert type(used_i) == list
    	assert complete == False

    	# Assert that an orthogonal projection has been added to Y, and that a
        # vector has been added to T. The dimensions of the dot product of
        # Y and T should equal X.
    	assert Y.shape[1] == T.shape[0] == 2
    	compression = np.dot(Y, T)
    	assert compression.shape == X.shape

    	# Definition of orthogonal matrix
    	print("Y.T.dot(Y)", Y.T.dot(Y))
    	assert np.isclose(np.linalg.det(Y.T.dot(Y).astype(float)), 1)
    	# assert np.testing.assert_array_almost_equal(Y.T.dot(Y), np.eye(2,2))


    def test_d1_done(self):
		"""
        Test mod_gramm_schmidt where d!=0 and the error requirement is met,
        ending the RBD algorithm.
        """

		# Setup parameters
		X,Y,T,d,e_cur, i,used_i, complete = matrix_to_attributes(test_image)
		Er = 0.00001

		# Complete 1 round of the RBD decomposition algorithm and update parameters.
		d1_parameters = check_error(*find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete)))
		X,Y,T,d,e_cur, Er, i,used_i, complete = d1_parameters

		# Ensure that the algorithm will think the decomposition is complete.
		Er = 50000

		# Apply gramm_schmidt at the beginning of stage d = 1
		gramm_schmidt_results = mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete)

		assert gramm_schmidt_results[-1] == True

    def test_find_new_vector(self):
        """
        Test that find_error returns new e_cur and i values.
        """

        # Setup parameters
        X,Y,T,d,e_cur, i,used_i, complete = matrix_to_attributes(self.test_image)
        Er = 0.0000000001

        # Complete a full iteration of RBD
        d1_parameters = check_error(*find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete)))

        # Take a snapshot of e_cur and i before running the find_error function
        e_cur_before_find_error = d1_parameters[4]
        i_before_find_error = d1_parameters[6]

        # Capture the results of find_error
        find_error_results = find_error(*mod_gramm_schmidt(*d1_parameters))
        e_cur_after_find_error = find_error_results[4]
        i_after_find_error = find_error_results[6]
        used_i = find_error_results[7]

        # Assert that the new e_cur and i values returned by find_error are distinct.
        assert e_cur_before_find_error != e_cur_after_find_error
        assert i_before_find_error != i_after_find_error
        assert i_after_find_error not in used_i

    def test_decomposition_complete(self):
        """
        Test case for when e_cur <= Er, signaling that the decomposition is done.
        """

        # Setup parameters
        X,Y,T,d,e_cur, i,used_i, complete = matrix_to_attributes(self.test_image)
        Er = np.inf

        # Capture results of RBD at the point before check_error is run.
        params_before_check_error = find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete))

        # Capture results after running check_error in a case that should return complete == True
        params_after_check_error = check_error(*params_before_check_error)

        # Test that check_error modifies complete to equal True.
        assert params_after_check_error[-1] == True != params_before_check_error[-1]

    def test_decomposition_incomplete(self):
        """
        Test case for check_error when the error requirement is not met.
        d increases to 1 for the next iteration of RBD.
        """

        # Setup parameters
        X,Y,T,d,e_cur, i,used_i, complete = matrix_to_attributes(test_image)
        Er = 0.000000000000001

        # Capture results of RBD at the point before check_error is run.
        params_before_check_error = find_error(*mod_gramm_schmidt(X,Y,T,d,e_cur, Er, i,used_i, complete))

        # Capture results after running check_error in a case that should return complete == False
        params_after_check_error = check_error(*params_before_check_error)

        # Test that d is increased by 1 after check_error.
        d_before_check_error = params_before_check_error[3]
        d_after_check_error = params_after_check_error[3]
        assert d_before_check_error < d_after_check_error
        assert d_after_check_error - d_before_check_error == 1

    def test_rbd_to_completion(self):
        """
        Test case that runs RBD with no cap on dmax and an error requirement
        that cannot be satisfied.
        """

        # Define dmax and Er
        dmax = self.test_image.shape[1]
        Er = -np.inf

        # Run RBD and capture the output
        X,Y,T,d,e_cur,Er,i,used_i,complete = rbd(self.test_image, dmax, Er)

        # Test that the compression YT has the same dimensions as the input matrix X
        compression = np.dot(Y,T)
        assert compression.shape == X.shape
        assert len(used_i) == d

        # Test that Y is orthogonal
        assert np.diagonal(Y.T.dot(Y)).all() == 1

    def test_rbd_early_break_d(self):
        """
        Test case that affirms that RBD quits at a user-defined dmax
        """
        # Define dmax and Er
        dmax = 5
        Er = -np.inf

        # Run RBD and capture the output
        X,Y,T,d,e_cur,Er,i,used_i,complete = rbd(self.test_image, dmax, Er)

        # Test that the compression YT has the same dimensions as the input matrix X
        compression = np.dot(Y,T)
        assert compression.shape == X.shape

        # Assert that RBD quit at the proper stage
        assert d == 6
        assert len(used_i) == d == dmax+1

        # Test that Y is orthogonal
        assert np.diagonal(Y.T.dot(Y)).all() == 1

    def test_rbd_early_break_e_cur(self):
        """
        Test case that sets the error threshold Er that will be satisfied by RBD.
        Affirms that RBD quits at that point.
        """

        # Define dmax and Er
        dmax = self.test_image.shape[1]
        Er = 255

        # Run RBD and capture the output
        X,Y,T,d,e_cur,Er,i,used_i,complete = rbd(self.test_image, dmax, Er)

        # Test that the compression YT has the same dimensions as the input matrix X
        compression = np.dot(Y,T)
        assert compression.shape == X.shape

        # Assert that RBD quit at the proper stage
        assert len(used_i) == d+1
        assert e_cur <= Er

        # Test that Y is orthogonal
        assert np.diagonal(Y.T.dot(Y)).all() == 1

    def test_command_line_interface(self):
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'reduced_basis_decomposition.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
