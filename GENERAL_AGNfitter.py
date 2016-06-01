from __future__ import division, unicode_literals

"""%%%%%%%%%%%%%%%%%

             GENERAL_AGNfitter.py

%%%%%%%%%%%%%%%%%%

This script contains all general non-physical functions created 
for a more easy and structured construction of other more complex functions.

It contains following functions:

*	Inter/Extrapolation functions

NearestNeighbourSimple2D(x, D, K)
NearestNeighbourSimple1D(x, D, K)
extrap1d

*	Dictionary construction functions (from Barak, Chrighton)

class adict(dict)
writetxt(fh, cols, sep=' ', names=None, header=None, overwrite=False, fmt_float='s')
loadobj(filename)
saveobj(filename, obj, overwrite=False)
puttext(x,y,text,ax, xcoord='ax', ycoord='ax', **kwargs)
distplot
dhist
autocorr
parse_config


"""



import numpy as np
import math
from math import exp,log,pi
import matplotlib.pyplot as plt
import pylab as pl
from numpy import random,argsort,sqrt, array
#from barak.utilities import adict
import re
import os
import time


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

INTERPOLATION FUNCTIONS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""


#==============================================================
#			 NEAREST NEIGHBOUR INTERPOLATION ALGORITHM : SIMPLE 2D
#==============================================================

def NearestNeighbourSimple2D(x, D, K):
 	""" find K nearest neighbours of data among D """
	#column of taus from data  (in case of bc: 'tau_galaxy', other:'1column' only)
	t = D[:,0]
	nt = len(np.unique(t))
 	# distances from the other points
	sqdx = (D[:,0] - x[0])**2
	# sorting first distances for first parameter
	idx = argsort(sqdx) 
	# taking from all data only the indices nearer for the first parameters
	idy_array = idx[:nt]
	#print idy_array
	sqdy = np.zeros(len(idy_array))
	# sorting 2nd distances for 2dn parameter
	for j in range(len(idy_array)):
	    	sqdy[j] = (D[idy_array[j],1] - x[1])**2
		idy = argsort(sqdy)
	# the index of the nearest par, if n are wanted:	idy[0, .., n]
	n=idy[0]
	nearest = np.array(idy_array[n])
	# return the indexes of K nearest neighbours
	return nearest

#==============================================================
#		              NEAREST NEIGHBOUR INTERPOLATION ALGORITHM : SIMPLE 1D
#==============================================================

def NearestNeighbourSimple1D(x, D, K):
 	""" find K nearest neighbours of data among D """
	#column of taus from data  (in case of bc: 'tau_galaxy', other:'1column' only)
	t = D

#	nt = len(np.unique(t))
 	# distances from the other points
	sqdx = (t - x)**2
	# sorting first distances for first parameter
#	idx = argsort(sqdx) 
#	n=idx[0]
#===
	n=np.argmin(sqdx)		
#===
	nearest = n
	# return the indexes of K nearest neighbours
	return nearest

#==============================================================
#		              EXTRA1PD - Extrapolation compatible to python inter1pd
#==============================================================

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:   
	    if ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])>0:
	        return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            else:
		return 0	
	elif x > xs[-1]:
	    if ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])>0:
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
		return 0	
	else:
	    if interpolator(x)>0:
            	return interpolator(x)
	    else: return 0
    def ufunclike(xs):
        return np.array(map(pointwise, array(xs)))

    return ufunclike


def frange(start, stop, step):
    r = start
    while r < stop:
         yield r
         r += step

"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 FUNCTIONS BORROWED FROM barak (Neil Chrighton c)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""




try:
    unicode
except NameError:
    import pickle
    unicode = basestring = str
    xrange = range
else:
    import cPickle as pickle


from textwrap import wrap
import sys, os
import numpy as np
from math import sqrt

class adict(dict):
    """ A dictionary with attribute-style access. It maps attribute
    access to the real dictionary."""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    # the following two methods allow pickling
    def __getstate__(self):
        """Prepare a state of pickling."""
        return self.__dict__.items()

    def __setstate__(self, items):
        """ Unpickle. """
        for key, val in items:
            self.__dict__[key] = val

    def __setitem__(self, key, value):
        return super(adict, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(adict, self).__getitem__(name)

    def __delitem__(self, name):
        return super(adict, self).__delitem__(name)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            # make sure existing methods are not overwritten by new
            # keys.
            return super(adict, self).__setattr__(key, value)
        else:
            return super(adict, self).__setitem__(key, value)

    __getattr__ = __getitem__

    def copy(self):
        """ Return a copy of the attribute dictionary.

        Does not perform a deep copy
        """
        return adict(self)

def writetxt(fh, cols, sep=' ', names=None, header=None, overwrite=False,
             fmt_float='s'):
    """ This is deprecated. Use `writetable()` with file type '.tbl'
    instead.

    Write data to a column-aligned text file.

    Structured array data written using this function can be read
    again using:

    >>> readtxt(filename, readnames=True)

    Parameters
    ----------
    fh :  file object or str
        The file to be written to.
    cols : structured array or a list of columns
        Data to be written.
    sep : str (' ')
        A string used to separate items on each row.
    names : list, string, False or None (None)
        Column names. Can be a comma-separated string of names. If
        False, do not print any names. If None and `cols` is a
        structured array, column names are the array field names.
    header : str (None)
        A header written before the data and column names.
    overwrite : bool (False)
        If True, overwrite an existing file without prompting.
    """
    # Open file (checking whether it already exists)
    if isinstance(fh, basestring):
        if not overwrite:
            while os.path.lexists(fh):
                c = raw_input('File %s exists, overwrite? (y)/n: ' % fh)
                if c == '' or c.strip().lower()[0] != 'n':
                    break
                else:
                    fh = raw_input('Enter new filename: ')
        fh = open(fh, 'wt')

    if isinstance(names, basestring):
        names = names.split(',')

    try:
        recnames = cols.dtype.names
    except AttributeError:
        pass
    else:
        if names not in (None, False):
            recnames = names
        cols = [cols[n] for n in recnames]
        if names is None:
            names = list(recnames)

    cols = [np.asanyarray(c) for c in cols]

    if names not in (None, False):
        if len(names) < len(cols):
            raise ValueError('Need one name for each column!')

    nrows = [len(c) for c in cols]
    if max(nrows) != min(nrows):
        raise ValueError('All columns must have the same length!')
    nrows = nrows[0]

    # Get the maximum field width for each column, so that the columns
    # will line up when printed. Also find the printing format for
    # each column.
    maxwidths = []
    formats = []
    for col in cols:
        dtype = col.dtype.str[1:]
        if dtype.startswith('S'):
            maxwidths.append(int(dtype[1:]))
            formats.append('s')
        elif dtype.startswith('i'):
            maxwidths.append(max([len('%i' % i) for i in col]))
            formats.append('i')
        elif dtype.startswith('f'):
            maxwidths.append(max([len(('%' + fmt_float) % i) for i in col]))
            formats.append(fmt_float)
        elif dtype.startswith('b'):
            maxwidths.append(1)
            formats.append('i')
        else:
            raise ValueError('Unknown column data-type %s' % dtype)

    if names not in (None, False):
        for i,name in enumerate(names):
            maxwidths[i] = max(len(name), maxwidths[i])

    fmt = sep.join(('%-'+str(m)+f) for m,f in zip(maxwidths[:-1], formats[:-1]))
    fmt += sep + '%' + formats[-1] + '\n'

    if names:
        fmtnames = sep.join(('%-' + str(m) + 's') for m in maxwidths[:-1])
        fmtnames += sep + '%s\n'

    # Write the header if it was given
    if header is not None:
        fh.write(header)

    if names:
        fh.write(fmtnames % tuple(names))
    for row in zip(*cols):
        fh.write(fmt % tuple(row))

    fh.close()
    return

def loadobj(filename):
    """ Load a python object pickled with saveobj."""
    if filename.endswith('.gz'):
        fh = gzip.open(filename, 'rb')
    else:
        fh = open(filename, 'rb')
    obj = pickle.load(fh)
    fh.close()
    return obj

def saveobj(filename, obj, overwrite=False):
    """ Save a python object to filename using pickle."""
    if os.path.lexists(filename) and not overwrite:
        raise IOError('%s exists' % filename)
    if filename.endswith('.gz'):
        fh = gzip.open(filename, 'wb')
    else:
        fh = open(filename, 'wb')
    pickle.dump(obj, fh, protocol=2)
    fh.close()



A4PORTRAIT = 8.3, 11.7
A4LANDSCAPE = 11.7, 8.3

def puttext(x,y,text,ax, xcoord='ax', ycoord='ax', **kwargs):
    """ Print text on an axis using axes coordinates."""
    if xcoord == 'data' and ycoord == 'ax':
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    elif xcoord == 'ax' and ycoord == 'data':
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    elif xcoord == 'ax' and ycoord == 'ax':
        trans = ax.transAxes
    else:
        raise ValueError("Bad keyword combination: %s, %s "%(xcoord,ycoord))
    return ax.text(x, y, str(text), transform=trans, **kwargs)

def distplot(vals, xvals=None, perc=(68, 95), showmean=False,
             showoutliers=True, color='forestgreen',  ax=None,
             logx=False, logy=False, negval=None, **kwargs):
    """
    Make a top-down histogram plot for an array of
    distributions. Shows the median, 68%, 95% ranges and outliers.

    Similar to a boxplot.

    Parameters
    ----------
    vals : sequence of arrays
        2-d array or a sequence of 1-d arrays.
    xvals : array of floats
        x positions.
    perc : array of floats  (68, 95)
        The percentile levels to use for area shading. Defaults show
        the 68% and 95% percentile levels; roughly 1 and 2
        sigma ranges for a Gaussian distribution.
    showmean : boolean  (False)
        Whether to show the means as a dashed black line.
    showoutliers : boolean (False)
        Whether to show outliers past the highest percentile range.
    color : mpl color ('forestgreen')
    ax : mpl Axes object
        Plot to this mpl Axes instance.
    logx, logy : bool (False)
        Whether to use a log x or y axis.
    negval : float (None)
        If using a log y axis, replace negative plotting values with
        this value (by default it chooses a suitable value based on
        the data values).
    """
    if any(not hasattr(a, '__iter__') for a in vals):
        raise ValueError('Input must be a 2-d array or sequence of arrays')

    assert len(perc) == 2
    perc = sorted(perc)
    temp = 0.5*(100 - perc[0])
    p1, p3 = temp, 100 - temp
    temp = 0.5*(100 - perc[1])
    p0, p4 = temp, 100 - temp
    percentiles = p0, p1, 50, p3, p4

    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)

    if xvals is None:
        xvals = np.arange(len(vals), dtype=float)


    # loop through columns, finding values to plot
    x = []
    levels = []
    outliers = []
    means = []
    for i in range(len(vals)):
        d = np.asanyarray(vals[i])
        # remove nans
        d = d[~np.isnan(d)]
        if len(d) == 0:
            # no data, skip this position
            continue
        # get percentile levels
        levels.append(np.percentile(d, percentiles))
        if showmean:
            means.append(d.mean())
        # get outliers
        if showoutliers:
            outliers.append(d[(d < levels[-1][0]) | (levels[-1][4] < d)])
        x.append(xvals[i])

    levels = np.array(levels)
    if logx and logy:
        ax.loglog([],[])
    elif logx:
        ax.semilogx([],[])
    elif logy:
        ax.semilogy([],[])

    if logy:
        # replace negative values with a small number, negval
        if negval is None:
            # guess number, falling back on 1e-5
            temp = levels[:,0][levels[:,0] > 0]
            if len(temp) > 0:
                negval = np.min(temp)
            else:
                negval = 1e-5

        levels[~(levels > 0)] = negval
        for i in range(len(outliers)):
            outliers[i][outliers[i] < 0] = negval
            if showmean:
                if means[i] < 0:
                    means[i] = negval

    ax.fill_between(x,levels[:,0], levels[:,1], color=color, alpha=0.2, edgecolor='none')
    ax.fill_between(x,levels[:,3], levels[:,4], color=color, alpha=0.2, edgecolor='none')
    ax.fill_between(x,levels[:,1], levels[:,3], color=color, alpha=0.5, edgecolor='none')
    if showoutliers:
        x1 = np.concatenate([[x[i]]*len(out) for i,out in enumerate(outliers)])
        out1 = np.concatenate(outliers)
        ax.plot(x1, out1, '.', ms=1, color='0.3')
    if showmean:
        ax.plot(x, means, 'k--')
    ax.plot(x, levels[:,2], 'k-', **kwargs)
    ax.set_xlim(xvals[0],xvals[-1])
    try:
        ax.minorticks_on()
    except AttributeError:
        pass

    return ax




def dhist(xvals, yvals, xbins=20, ybins=20, ax=None, c='b', fmt='.', ms=1,
          label=None, loc='right,bottom', xhistmax=None, yhistmax=None,
          histlw=1, xtop=0.2, ytop=0.2, chist=None, **kwargs):
    """ Given two set of values, plot two histograms and the
    distribution.

    xvals,yvals are the two properties to plot.  xbins, ybins give the
    number of bins or the bin edges. c is the color.
    """

    if chist is None:
        chist = c
    if ax is None:
        pl.figure()
        ax = pl.gca()

    loc = [l.strip().lower() for l in loc.split(',')]

    if ms is None:
        ms = default_marker_size(fmt)

    ax.plot(xvals, yvals, fmt, color=c, ms=ms, label=label, **kwargs)
    x0,x1,y0,y1 = ax.axis()

    if np.__version__ < '1.5':
        x,xbins = np.histogram(xvals, bins=xbins, new=True)
        y,ybins = np.histogram(yvals, bins=ybins, new=True)
    else:
        x,xbins = np.histogram(xvals, bins=xbins)
        y,ybins = np.histogram(yvals, bins=ybins)

    b = np.repeat(xbins, 2)
    X = np.concatenate([[0], np.repeat(x,2), [0]])
    Xmax = xhistmax or X.max()
    X = xtop * X / Xmax
    if 'top' in loc:
        X = 1 - X
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(b, X, color=chist, transform=trans, lw=histlw)

    b = np.repeat(ybins, 2)
    Y = np.concatenate([[0], np.repeat(y,2), [0]])
    Ymax = yhistmax or Y.max()
    Y = ytop * Y / Ymax
    if 'right' in loc:
        Y = 1 - Y
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot(Y, b, color=chist, transform=trans, lw=histlw)

    ax.set_xlim(xbins[0], xbins[-1])
    ax.set_ylim(ybins[0], ybins[-1])
    if pl.isinteractive():
        pl.show()

    return ax, dict(x=x, y=y, xbinedges=xbins, ybinedges=ybins)




def autocorr(x, maxlag=300):
    """ Find the autocorrelation of x.

    The mean is subtracted from x before correlating. Correlation
    values are calculated in offset steps from 0 up to maxlag.
    """
    dot = np.dot
    maxlag = min(maxlag, len(x)-1)
    x = np.asanyarray(x)
    x = x - x.mean()
    a = [1]
    for k in xrange(1, maxlag):
        v1 = dot(x[:-k], x[k:])
        v2 = dot(x[:-k], x[:-k])
        v3 = dot(x[k:], x[k:])
        a.append(v1 / sqrt(v2*v3))

    return a


def parse_config(filename, defaults={}):
    """ Read options for a configuration file. It does some basic type
    conversion (boolean, float or string).

    Parameters
    ----------
    filename : str or file object
      The configuration filename or a file object.
    defaults : dict
      A dictionary with default values for options.

    Returns
    -------
    d : dictionary
      The options are returned as a dictionary that can also be
      indexed by attribute.

    Notes
    -----
    Ignores blank lines, lines starting with '#', and anything on a
    line after a '#'. The parser attempts to convert the values to
    int, float or boolean, otherwise they are left as strings.

    Sample format::

     # this is the file with the line list
     lines = lines.dat
     x = 20
     save = True    # save the data
    """
    cfg = adict()

    if isinstance(filename, basestring):
        fh = open(filename, 'rb')
    else:
        fh = filename

    for row in fh:
        row = row.decode('utf-8')
        if not row.strip() or row.lstrip().startswith('#'):
            continue
        option, value = [r.strip() for r in row.split('#')[0].split('=', 1)]
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                elif value == 'None':
                    value = None

        if option in cfg:
            raise RuntimeError("'%s' appears twice in %s" % (option, filename))
        cfg[option] = value

    for key,val in defaults.items():
        if key not in cfg:
            cfg[key] = val

    fh.close()
    return cfg



def get_fig_axes(nrows, ncols, npar, width=13):
    fig = pl.figure(figsize=(width, width*1.6))#*nrows/ncols))    
    fig.subplots_adjust(hspace=0.9)
    axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(npar)]
    return fig, axes



def get_nrows_ncols(nplots, prefer_rows=True):
    """ Get the number of rows and columns to plot a given number of plots.

    Parameters
    ----------
    nplots : int
      Desired number of plots.

    Returns
    -------
    nrows, ncols : int
    """
    nrows = max(int(np.sqrt(nplots)), 1)
    ncols = nrows
    while nplots > (nrows * ncols):
        if prefer_rows:
            nrows += 1
        else:
            ncols += 1

    return nrows, ncols

