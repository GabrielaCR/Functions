`
`import numpy as np
import matplotlib.pyplot as plt

from GENERAL_AGNfitter import NearestNeighbourSimple2D, NearestNeighbourSimple1D, extrap1d
from scipy.interpolate import interp1d, UnivariateSpline
from numpy import array

from math import sqrt

import MODEL_AGNfitter as model
import PLOTandWRITE_AGNfitter_mock as plot_write2
from DATA_AGNfitter import DATA, NAME, REDSHIFT, DISTANCE
import DICTIONARIES_AGNfitter as dicts


def MOCKdata (z, dict_modelfiles, dict_modelfluxes, parameters):

	all_tau, all_age, all_nh, all_irlum, filename_0_galaxy, filename_0_starburst, filename_0_torus = dict_modelsfiles
	STARBURSTFdict , BBBFdict, GALAXYFdict, TORUSFdict, EBVbbb_array, EBVgal_array = dict_modelfluxes
	tau, age, nh, irlum, SB ,BB, GA,TO, BBebv, GAebv = parameters


	SB_filename = model.pick_STARBURST_template(irlum, filename_0_starburst, all_irlum)
	GA_filename = model.pick_GALAXY_template(tau, age, filename_0_galaxy, all_tau, all_age)
	TOR_filename = model.pick_TORUS_template(nh, all_nh, filename_0_torus)
	BB_filename = model.pick_BBB_template()

	EBV_bbb_0 = model.pick_EBV_grid(EBVbbb_array, BBebv)
	EBV_bbb = (  str(int(EBV_bbb_0)) if  float(EBV_bbb_0).is_integer() else str(EBV_bbb_0))
	EBV_gal_0 = model.pick_EBV_grid(EBVgal_array,GAebv)
	EBV_gal = (  str(int(EBV_gal_0)) if  float(EBV_gal_0).is_integer() else str(EBV_gal_0)) 

	if (GA_filename, EBV_gal) in GALAXYFdict:
		bands, gal_Fnu = GALAXYFdict[GA_filename, EBV_gal]
	else:	
		print 'Error: Dictionary does not contain key of ', GA_filename, EBV_gal, ' or the E(B-V) grid or the DICTIONARIES_AGNfitter file does not match when the one used in PARAMETERSPACE_AGNfitter/ymodel.py'

	if SB_filename in STARBURSTFdict:	
		bands, sb_Fnu= STARBURSTFdict[SB_filename]
	else:
		print 'Error: Dictionary does not contain key'+SB_filename+'. all STARBURST files or the E(B-V) grid.'

	if (BB_filename, EBV_bbb) in BBBFdict:
		bands, bbb_Fnu = BBBFdict[BB_filename, EBV_bbb]	
	else:
		print'Error: Dictionary does not contain key: '+ BB_filename, EBV_bbb +'or the E(B-V) grid of the DICTIONARIES_AGNfitter file does not match when the one used in PARAMETERSPACE_AGNfitter/ymodel.py'

	if TOR_filename in TORUSFdict:
		bands, tor_Fnu= TORUSFdict[TOR_filename]
	else:
		print 'Error: Dictionary does not contain TORUS file:'+TOR_filename

	NormalizationFLux = 1e-28 # This is the flux to which the models are normalized to  have comparable NORM factors
	sb_Fnu_norm = sb_Fnu / 1e18 
	bbb_Fnu_norm = bbb_Fnu / 1e58
	gal_Fnu_norm = gal_Fnu/ 1e12
	tor_Fnu_norm = tor_Fnu/  1e-42 

	# Sum components
	# ---------------------------------------------------------------------------------------------------------------------------------------------------------#
	lum =    10**(SB)* sb_Fnu_norm      +     10**(BB)*bbb_Fnu_norm    +     (10**GA)*gal_Fnu_norm     +     10**TO *tor_Fnu_norm 

	#-----------------------------------------------------------------------------------------------------------------------------------------------------------
	mockfluxes_array = lum.reshape((np.size(lum),))	

	return bands, mockfluxes_array


def MOCKdata_chain():

	"""
	parameters are given as
	Tau, Age, N_h, irlum, SB, BB, GA, TO, BBebv, GAebv

	# ---------------|-------------|-----|-------|--------|------|-------|------|------|------|
    P.names =  'tau',  	  'age',	'nh', 'irlum', 'SB',   'BB',	 'GA',	'TO','BBebv','GAebv'
    # ---------------|-------------|-----|-------|--------|------|-------|------|------|------|
    P.min = 	0 , 	10**6,	     21,	7, 		-20,  	 -20,	 -20,	 -20,   0, 	 0
    P.max = 	3.5,	max_age(z),	 25,	15, 	 20,	  20, 	  20,  	  20,   1, 	 0.5
    # ---------------|-------------|-----|-------|--------|------|-------|------|------|------|


	"""
	#	1. Quiescent old galaxy, constant SFH	
	z1 = 1.5
	parameters1 = np.array([1, 1e10, 21, 10, 2, -20, -0.5, -20, 0, 0  ])
	#	2. Quiescent old galaxy, decaying SFH	

	z1 = 1.5
	parameters2 = np.array([0.3, 1e10, 21, 10, 2, -20, -0.5, -20, 0, 0  ])

	#	3. Quiescent old galaxy, steep SFH	

	z1 = 1.5
	parameters3 = np.array([3, 1e10, 21, 10, 2, -20, -0.5, -20, 0 ,0 ])

	#	4. Quiescent young galaxy, constant SFH	

	z1 = 1.5
	parameters4 = np.array([1, 1e7, 21, 10, 2, -20, -0.5, -20, 0 ,0 ])

	#	5. Quiescent young galaxy, decaying SFH	

	z1 = 1.5
	parameters5 = np.array([0.3, 1e7, 21, 10, 2, -20, -0.5, -20, 0  ,0])

	#	6. Quiescent young galaxy, steep SFH	

	z1 = 1.5
	parameters6 = np.array([3, 1e7, 21, 10, 2, -20, -0.5, -20, 0  ,0])

	#	7. Starbursting galaxy

	z1 = 1.5
	parameters7 = np.array([1, 1e8, 21, 15, 0.1, -20, -0.5, -20, 0 , 0 ])

	#	Seyfert 1 0<z<0.5

	z2 = 0.2
	parameters8 = np.array([1, 1e9, 21, 15, -0.5, 1.5, 0.1, 0.1, 0 , 0 ])

	#	Seyfert 1 0<z<0.5 reddened

	z2 = 0.2
	parameters9 = np.array([1, 1e9, 21, 15, -0.5, 1.5, 0.1, 0.1, 0.6 , 0 ])

	#Seyfert 2 0<z 0.5

	z2 = 0.2
	parameters10 = np.array([1, 1e9, 24, 15, -0.5, 0.1, 0.2, 3.5, 0 , 0 ])

	#Seyfert 2 0<z 0.5 reddened

	z2 = 0.2
	parameters11 = np.array([1, 1e9, 24, 15, -0.5, 0.1, 0.2, 3.5, 0.6 , 0 ])

	#quasar 1 z=3

	z3 = 3
	parameters12 = np.array([1, 1e10, 21, 15, 0.1, 2, 1, 2, 0 , 0 ])

	#quasar 1 z=3 reddened

	z3 = 3
	parameters13 = np.array([1, 1e10, 21, 15, 0.1, 2, 1, 2, 0.6 , 0 ])

	#quasar 2 z=3 

	z3 = 3
	parameters14 = np.array([1, 1e10, 24, 15, 0.1, 2, 2, 4, 0 , 0 ])

	#quasar 2 z=3 
	z3 = 3
	parameters15 = np.array([1, 1e10, 24, 15, 0.1, 2, 2, 4, 0.6 , 0 ])

	mock_chain = np.vstack((parameters1, parameters2, parameters3, parameters4, parameters5, parameters6, parameters7, parameters8, parameters9, parameters10, parameters11, parameters12, parameters13, parameters14, parameters15))

	zs = np.vstack((z1,z1,z1,z1,z1,z1,z1,z2,z2,z2,z2,z3,z3,z3,z3))
	print 'shape of mock chain:', np.shape(mock_chain)

	return mock_chain, zs



def MOCKdata_testplot(nr):

	catalog = '/Users/Gabriela/Codes/AGNfitter/data/catalog_type2_hardsel.dat'
	sourceline = 1
	chain, zs = MOCKdata_chain()
	chain2 = np.vstack((  chain[9], chain[0]))

	z= zs[0]
	source = NAME(catalog,sourceline)
	data_nus, ydata, ysigma = DATA(catalog, sourceline)
	path = '/Users/Gabriela/Codes/AGNfitter/'
	filterdict = dicts.filter_dictionaries('COSMOS1', path)
	dict_modelsfiles = dicts.arrays_of_modelparsandfiles(path)

	all_model_nus, FLUXES4plotting = plot_write2.fluxes_arrays(data_nus, catalog, sourceline, dict_modelsfiles, filterdict, chain2, nr, path)

	fig = plot_write2.PLOT_SED_manyrealizations(source, data_nus, ydata, ysigma, z, all_model_nus, FLUXES4plotting, nr)
	plt.savefig('mocks_test.pdf', format = 'pdf')#, dpi = 900)

	return 'done'


def MOCKdata_construct_catalog (mock_type, catalog, sourceline):

	mock_chain, zs = MOCKdata_chain()

	parameters = mock_chain[mock_type]
	z = zs[mock_type]

	dictionarypath = '/Users/Gabriela/Codes/AGNfitter/models/COSMOS_modelsdict_MacOSX'
	path = '/Users/Gabriela/Codes/AGNfitter/'

	print 'Openning dictionary from'
	print dictionarypath

	COSMOS_modelsdict = shelve.open(dictionarypath)
	dict_modelfiles = dicts.arrays_of_modelparsandfiles(path)

	z_array_in_dict = np.arange(0.1,4.1,0.01) #Has to be the same as used to create the general dict
	z_key = str( z_array_in_dict[NearestNeighbourSimple1D(z, z_array_in_dict , 1)] 	)

	print 'Downloading model dictionary. '
	dict_modelfluxes = COSMOS_modelsdict[z_key]
	COSMOS_modelsdict.close

	mockfluxes = MOCKdata (z, dict_modelfiles, dict_modelfluxes, parameters)

	return mockfluxes




