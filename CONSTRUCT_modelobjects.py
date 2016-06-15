import numpy as np
import math

import time
import cPickle
import shelve
from astropy import units as u 



	


def STARBURST_read (fn):

	#reading
	c = 2.997e10
	c_Angst = 3.34e-19 #(1/(c*Angstrom)
	
	dh_wl_rest, dh_Flambda =  np.loadtxt(fn, usecols=(0,1),unpack= True)
	dh_wl = dh_wl_rest 
	dh_nu_r = np.log10(c / (dh_wl * 1e-8)) 
	dh_Fnu = dh_Flambda * (dh_wl**2. )* c_Angst

	#reverse , in order to have increasing frequency
	dh_nus= dh_nu_r[::-1]
	dh_Fnu = dh_Fnu[::-1]

	return dh_nus, dh_Fnu

def TORUS_read(tor_file):

	distance= 1e27

	tor_nu_rest, tor_nuLnu = np.loadtxt(tor_file, skiprows=0, usecols=(0,1),unpack= True)
	tor_Lnu = tor_nuLnu / 10**(tor_nu_rest)	
	tor_Fnu = tor_Lnu /(4. * np.pi * distance**2.) 

	return tor_nu_rest, tor_Fnu 

def BBB_read(bbb_file):

	bbb_nu_log_rest, bbb_nuLnu_log = np.loadtxt(bbb_file, usecols=(0,1),unpack= True)
	bbb_nu_exp = 10**(bbb_nu_log_rest) 
	bbb_nu = np.log10(10**(bbb_nu_log_rest) )
	bbb_nuLnu= 10**(bbb_nuLnu_log)
	bbb_Lnu = bbb_nuLnu / bbb_nu

	bbb_x = bbb_nu
	bbb_y =	bbb_nuLnu  / bbb_nu_exp

	return bbb_x, bbb_y






class MODEL:

	def __init__(self, physcomponent):
		self.component = physcomponent
		self.path ='/Users/Gabriela/Desktop/AGNfitter_newcode/AGNfitter_oldversion/'

	def build(self):

		wave_array = []
		sed_array = []

		if self.component=='starburst':

			irlum1, all_filenames1 = np.loadtxt(self.path+'models/STARBURST/DALE.list' , usecols=(2,0), dtype = 'S', unpack= True)
			irlum2, all_filenames2 = np.loadtxt(self.path+'models/STARBURST/CHARY_ELBAZ.list' , usecols=(2,0), dtype = 'S',unpack= True)

			self.irlum = np.concatenate((irlum1.astype(float), irlum2.astype(float)))
			all_filenames = np.concatenate((all_filenames1, all_filenames2))

			for i in range(len(all_filenames)):
				wave, SED = STARBURST_read(self.path+'models/STARBURST/'+all_filenames[i])
				wave_array.append(wave)
				sed_array.append(SED)

			self.wave = np.asarray(wave_array)
			self.SED = np.asarray(sed_array)



		if self.component== 'torus':

			tor_list = self.path +'models/TORUS/torus_templates_list.dat'
			nh, all_filenames= np.loadtxt(tor_list , usecols=(0,1), unpack=True, dtype = ('S'))
			self.nh = nh.astype(float)
			for i in range(len(all_filenames)):
				wave, SED = TORUS_read(self.path+all_filenames[i])
				wave_array.append(wave)
				sed_array.append(SED)

			self.wave = np.asarray(wave_array)
			self.SED = np.asarray(sed_array)

		if self.component== 'bbb':
		
			filename= 	self.path + 'models/BBB/richardsbbb.dat'
			self.wave, self.SED = BBB_read(filename)



path ='/Users/Gabriela/Desktop/AGNfitter_newcode/AGNfitter_oldversion/'



SB = MODEL('starburst')
SB.build()
f = open(path + 'models/STARBURST/dalehelou_charyelbaz_v1.pickle', 'wb')
cPickle.dump(SB, f, protocol=2)
f.close()


TO = MODEL('torus')
TO.build()
f1 = open(path + 'models/TORUS/silva_v1.pickle', 'wb')
cPickle.dump(TO, f1, protocol=2)
f1.close()

BB = MODEL('bbb')
BB.build()
f2 = open(path + 'models/BBB/richards.pickle', 'wb')
cPickle.dump(BB, f2, protocol=2)
f2.close()




		
