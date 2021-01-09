#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Gabe Schamberg
Date: Jan 5, 2021
Description: Functions for generating a multivariate Gaussian System
"""

import numpy as np
import scipy

def generate_system(system='random',dm=5,dx=5,dy=5):
	"""
	Generate appropriate gain and covariance matrices

	Param
	-----------
	system : str
		Specify which kind of system to generate. Either 'random' for
		a random covariance matrix or one of the other special
		examples

	dm : int
		dimensionality of M vector
	
	dx : int
		dimensionality of X vector
	
	dy : int
		dimensionality of Y vector

	Returns
	-------
	hx : np.ndarray [dx,dm]
		Channel gain matrix for M->X (i.e. X|M has mean hx.dot(M))

	hy : np.ndarray [dy,dm]
		Channel gain matrix for M->Y

	hxy : np.ndarray [dx+dy,m]
		Channel gain matrix for M->(X,Y)

	sigx : np.ndarray [dx,dx]
		Channel covariance matrix for M->X (i.e. X|M has cov sigx)

	sigy : np.ndarray [dy,dy]
		Channel covariance matrix for M->Y

	sigxy : np.ndarray [dx+dy,dx+dy]
		Channel covariance matrix for M->(X,Y)

	covxy : np.ndarray [dx+dy,dx+dy]
		Covariance matrix for (X,Y)

	sigm : np.ndarray [dm,dm]
		Covariance matrix for M

	"""
	systems = ['random',
			   'both fully unique',
			   'only one unique',
			   'one is very unique',
			   'some of each',
			   'univariate']

	assert system in systems, 'system must be on of {systems}'

	if system == 'random':
		# generate the full covariance matrices and submatrices
		cov = np.random.randn(dm+dx+dy,dm+dx+dy)
		cov = np.dot(cov,cov.T)
		covm = cov[:dm,:dm]
		covx = cov[dm:dm+dx,dm:dm+dx]
		covy = cov[dm+dx:dm+dx+dy,dm+dx:dm+dx+dy]
		covxy = cov[dm:dm+dx+dy,dm:dm+dx+dy]
		covx_m = cov[:dm,dm:dm+dx].T
		covy_m = cov[:dm,dm+dx:dm+dx+dy].T
		covxy_m = cov[:dm,dm:dm+dx+dy].T
		# channel gain matrices (using multivariate conditional mean)
		hx = covx_m.dot(np.linalg.inv(covm))
		hy = covy_m.dot(np.linalg.inv(covm))
		hxy = covxy_m.dot(np.linalg.inv(covm))
		# channel covariance matrices (using mvar conditional covariance)
		sigm = covm
		sigx = covx - covx_m.dot(np.linalg.inv(covm)).dot(covx_m.T)
		sigy = covy - covy_m.dot(np.linalg.inv(covm)).dot(covy_m.T)
		sigxy = covxy - covxy_m.dot(np.linalg.inv(covm)).dot(covxy_m.T)

	# would expect both to be equally deficient
	elif system == 'both fully unique':
		dm = 2
		dx = 1
		dy = 1

		# X is M1 plus noise
		hx = np.array([1,0]).reshape(1,2)
		# Y is M2 plus noise
		hy = np.array([0,1]).reshape(1,2)

		sigm = np.eye(dm)+0.3*np.eye(dm).T
		sigx = 0.0001*np.eye(dx)
		sigy = 0.0001*np.eye(dy)

	# would expect one to be deficient but not the other
	elif system == 'only one unique':
		dm = 2
		dx = 2
		dy = 2

		# make both X and Y exact copies of M
		hx = np.eye(2)
		hy = np.eye(2)

		sigm = np.eye(dm)
		# but X is noisier
		sigx = 4*np.eye(dx)
		sigy = 0.01*np.eye(dy)
		
	# would expect one to be deficient but want to ensure it is also fully redundant
	elif system == 'one is very unique':
		dm = 10
		dx = 1
		dy = dm

		# X only gets dx of the dm dimensions
		hx = np.hstack((np.eye(dx),np.zeros((dx,dm-dx))))
		# Y gets a copy of everything
		hy = np.eye(dy)

		sigm = np.eye(dm)
		sigx = 1*np.eye(dx)
		sigy = 1*np.eye(dy)
		
	# would expect one to be deficient but not the other
	elif system == 'some of each':
		dm = 3
		dx = 2
		dy = 2

		# make both X and Y get some of dim 1 and one of the other 2 dimensions
		hx = np.array([[1,0,0],[0,0,1]])
		hy = np.array([[np.random.uniform(-1,1),0,0],[0,1,0]])

		sigm = np.eye(dm)
		sigx = np.eye(dx)
		sigy = np.eye(dy)
		
	# can compare with Barrett results
	elif system == 'univariate':
		dm = 1
		dx = 1
		dy = 1

		hx = np.random.uniform(-1,1,size=(dx,dm))
		hy = np.random.uniform(-1,1,size=(dx,dm))
	   
		sigm = np.eye(dm)
		sigx = np.eye(dx) - hx.dot(sigm).dot(hx.T)
		sigy = np.eye(dy) - hy.dot(sigm).dot(hy.T)

	if system != 'random':
		hxy = np.vstack((hx,hy))
		sigxy = np.bmat([[sigx,np.zeros((dx,dy))],[np.zeros((dy,dx)),sigy]])
		covx = sigx + hx.dot(sigm).dot(hx.T)
		covy = sigy + hy.dot(sigm).dot(hy.T)
		covxy = np.bmat([[covx,np.zeros((dx,dy))],[np.zeros((dy,dx)),covy]])


		
	# whiten
	hx = scipy.linalg.sqrtm(np.linalg.inv(sigx)).dot(hx)
	sigx = np.eye(dx)
	hy = scipy.linalg.sqrtm(np.linalg.inv(sigy)).dot(hy)
	sigy = np.eye(dy)

	return hx,hy,hxy,sigx,sigy,sigxy,covxy,sigm