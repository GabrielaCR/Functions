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
from astropy.table import Table
from astropy.io import fits
import shelve
from functions.GENERAL_AGNfitter import NearestNeighbourSimple1D
import functions.DICTIONARIES_AGNfitter as dicts


class DATA:

	"""
	Class DATA
	input: catalogname, sourceline
	bugs: Not ready to read FITS yet.

	"""

	def __init__(self, cat, sourceline):
		self.cat = cat
		self.sourceline = sourceline
		self.catalog = cat['filename']
		self.path = cat['path']
		self.dict_path = cat['dict_path']
		self.output_folder = cat['output_folder']

	def PROPS(self):


		if self.cat['filetype'] == 'ASCII': 

			data = open(self.catalog, 'r') 
			header = data.readline()
			for i in range(0, self.sourceline):
				header = data.readline()
			line = data.readline()
			column = np.array(line.strip().split())


			self.name = column[self.cat['name']]
			self.z = float(column[self.cat['redshift']])


			freq_wl_cat = [column[c] * self.cat['freq/wl_unit'] for c in self.cat['freq/wl_list']]

			## If columns with flags exist
			if self.cat['ndflag_bool'] == True: 
				if self.cat['freq/wl_format']== 'frequency' :
					nus = [freq_wl_cat[i].to(u.Hz) for i in range(len(freq_wl_cat))].asarray()
				if self.cat['freq/wl_format']== 'wavelength' :
					nus = [freq_wl_cat[i].to(u.Hz, equivalencies=u.spectral()) for i in range(len(freq_wl_cat))].asarray()


				flux_cat = [ca*self.cat['flux_unit'] for ca in  column[self.cat['flux_list']] ]
				fluxes = np.array([flux_cat[i].to(u.erg/ u.s/ (u.cm)**2 / u.Hz) for i in range(len(freq_wl_cat))])

				fluxerr_cat = [ce *self.cat['flux_unit'] for ce in column[self.cat['fluxerr_list']]]
				fluxerrs = np.array([fluxerr_cat[i].to(u.erg/ u.s/(u.cm)**2/u.Hz) for i in range(len(freq_wl_cat))])

				ndflag_cat = column[self.cat['ndflag_list']]

			## If NO columns with flags exist
			elif self.cat['ndflag_bool'] == False:
				if self.cat['freq/wl_format']== 'frequency' :
					nus = np.log10(np.array([freq_wl_cat[i].to(u.Hz) for i in range(len(freq_wl_cat))]))
				if self.cat['freq/wl_format']== 'wavelength' :
					nus = np.log10(np.array([freq_wl_cat[i].to(u.Hz, equivalencies=u.spectral()) for i in range(len(freq_wl_cat))]))

				flux_cat = [ca*self.cat['flux_unit'] for ca in  column[self.cat['flux_list']] ]
				fluxes = np.array([flux_cat[i].to(u.erg/ u.s/ (u.cm)**2 / u.Hz) for i in range(len(freq_wl_cat))])

				fluxerr_cat = [ce *self.cat['flux_unit'] for ce in column[self.cat['fluxerr_list']]]
				fluxerrs = np.array([fluxerr_cat[i].to(u.erg/ u.s/(u.cm)**2/u.Hz) for i in range(len(freq_wl_cat))])

				ndflag_cat = np.ones(len(flux_cat))
				ndflag_cat[fluxes< 0] = 0.


		elif self.cat['filetype'] == 'FITS': 

			data =Table.read(self.cat['filename'],hdu=1)
			column = data[self.sourceline]
			print data.colnames


		## Sort in order of frequency
		self.nus = nus[nus.argsort()]
		self.fluxes = fluxes[nus.argsort()]
		self.fluxerrs = fluxerrs[nus.argsort()]
		self.ndflag = ndflag_cat[nus.argsort()]


	def DICTS(self, mc):

		self.PROPS()

		COSMOS_modelsdict = shelve.open(self.dict_path)
		z_array_in_dict = np.arange(0.1,4.1,0.05)             
		z_key= str( z_array_in_dict[NearestNeighbourSimple1D( self.z, z_array_in_dict , 1)]   )


		COSMOS_modelsdict = shelve.open(self.dict_path)
		self.dict_modelsfiles = dicts.arrays_of_modelparsandfiles(self.path)
		self.filterdict = dicts.filter_dictionaries(mc['Bandset'], self.path)           
		self.dict_modelfluxes = COSMOS_modelsdict[z_key]
		COSMOS_modelsdict.close


