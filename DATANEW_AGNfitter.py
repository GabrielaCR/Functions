"""%%%%%%%%%%%%%%%%%

            DATA_AGNFitter.py

%%%%%%%%%%%%%%%%%%

This script contains the class data, which administrate the catalog given by the user. 
Functions contained here are the following:

DATA
NAME
DISTANCE
REDSHIFT

"""
import numpy as np
from math import exp,log,pi, sqrt
import matplotlib.pyplot as plt
from numpy import random,argsort,sqrt
import time
from scipy.integrate import quad, trapz
from astropy import constants as const
from astropy import units as u

class DATA:

	def __init__(self, cat, sourceline):
		self.cat = cat
		self.sourceline = sourceline

	def DATA(self, cat, sourceline):

		if cat['filetype'] = 'ASCII': 

			

		if cat['filetype'] = 'FITS': 

		(1000 * u.nm).to(u.Hz, equivalencies=u.spectral())

	def NAME(self):
		print "0"

	def REDSHIFT (self):



		print "0"

	def DISTANCE (self):
		print "0"
