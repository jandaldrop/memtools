from __future__ import print_function
import numpy as np
import pandas as pd
import os.path
from correlation import *
from memtools import *
from scipy import interpolate
from scipy.integrate import cumtrapz
from flist import *
import ckernel


import sys
if sys.version_info < (3,0):
    print("Import python3 like zip function.")
    from itertools import izip as zip

class Igle(object):
    def __init__(self,xva_arg,saveall=True,prefix="",verbose=True,kT=2.494,trunc=1.,__override_time_check__=False,initial_checks=True,first_order=False):
        """
xva_arg should be either a pandas timeseries
or a listlike collection of them. Set xva_arg=None for load mode.
        """
        if xva_arg is not None:
            if isinstance(xva_arg,pd.DataFrame):
                self.xva_list=[xva_arg]
            else:
                self.xva_list=xva_arg
            if isinstance(self.xva_list,flist) and not initial_checks:
                print("WARNING: Consider setting initial_checks to False.")
            if initial_checks:
                for xva in self.xva_list:
                    for col in ['t','x','v','a']:
                        if col not in xva.columns:
                            raise Exception("Please provide txva data frame, or an iterable collection (i.e. list) of txva data frames. And not some other shit.")
        else:
            self.xva_list=None

        self.saveall=saveall
        self.prefix=prefix
        self.verbose=verbose
        self.kT=kT
        self.first_order=first_order

        # filenames
        self.corrsfile="corrs.txt"
        self.interpfefile="interp-fe.txt"
        self.histfile="fe-hist.txt"
        self.ucorrfile="u-corr.txt"
        self.kernelfile="kernel.txt"
        self.kernelfile_1st="kernel_1st.txt"

        self.corrs=None
        self.ucorr=None
        self.mass=None
        self.fe_spline=None
        self.fe=None
        self.per=False

        self.x0_fe=None
        self.x1_fe=None

        if self.xva_list is None:
            return

        # processing input arguments
        self.weights=np.array([xva.shape[0] for xva in self.xva_list],dtype=int)
        self.weightsum=np.sum(self.weights)

        if self.verbose:
            print("Found trajectories with the following legths:")
            print(self.weights)

        if initial_checks:
            lastinds=np.array([xva.index[-1] for xva in self.xva_list],dtype=int)
            smallest=np.min(lastinds)
            if smallest < trunc:
                if self.verbose:
                    print("Warning: Found a trajectory shorter than the argument trunc. Override.")
                trunc=smallest
        self.trunc=trunc

        if initial_checks and not __override_time_check__:
            sxva=self.xva_list[np.argmin(self.weights)]
            for xva in self.xva_list:
                if xva is not sxva:
                    if not sxva[sxva.index < trunc].index.equals(xva[xva.index < trunc].index):
                        raise Exception("Index mismatch.")

    def set_periodic(self,x0=-180,x1=180):
        if self.verbose:
            if not self.fe_spline is None:
                print("Reset free energy.")
        self.fe_spline=None
        self.per=True
        self.x0=x0
        self.x1=x1

    def compute_mass(self):
        if self.verbose:
            print("Calculate mass...")
            print("Use kT:", self.kT)

        if self.corrs["vv"] is None:
            v2sum=0.
            for i,xva in enumerate(self.xva_list):
                v2sum+=(xva["v"]**2).mean()*self.weights[i]
            v2=v2sum/self.weightsum
            self.mass=self.kT/v2
        else:
            self.mass=self.kT/self.corrs["vv"].iloc[0]

        if self.verbose:
            print("Found mass:", self.mass)

    def compute_fe(self, bins="auto", fehist=None, _dont_save_hist=False):
        '''Computes the free energy. If you run into memory problems, you can provide an histogram.'''
        if self.verbose:
            print ("Calculate histogram...")

        if fehist is None:
            if self.per:
                if type(bins) is str:
                    raise Exception("Strings not supported for periodic data.")
                if type(bins) is int:
                    bins=np.linspace(self.x0,self.x1,bins)

            fehist=np.histogram(np.concatenate([xva["x"].values for xva in self.xva_list]),bins=bins)

        if self.verbose:
            print("Number of bins:",len(fehist[1])-1)
            print ("Interpolate... (ignore p=0!)")
            if self.per:
                print("Assume PERIODIC data.")
            else:
                print("Assume NON-PERIODIC data.")

        xfa=(fehist[1][1:]+fehist[1][:-1])/2.

        pf=fehist[0]
        xf=xfa[np.nonzero(pf)]
        fe=-np.log(pf[np.nonzero(pf)])


        if self.per:
            if xf[0] != xfa[0]:
                raise Exception("No counts at lower edge of periodic boundary currently not supported.")
            xf=np.append(xf,xf[-1]+(xfa[-1]-xfa[-2]))
            fe=np.append(fe,0.)
            assert(xf[-1]-xf[0]==self.x1-self.x0)
            self.x0_fe=xf[0]
            self.x1_fe=xf[-1]

        self.fe_spline=interpolate.splrep(xf, fe, s=0, per=self.per)
        self.fe=pd.DataFrame({"F":fe},index=xf)

        if self.saveall:
            dxf=xf[1]-xf[0]
            xfine=np.arange(xf[0],xf[-1],dxf/10.)
            yi_t=interpolate.splev(xfine, self.fe_spline)
            yider_t=interpolate.splev(xfine, self.fe_spline, der=1)
            np.savetxt(self.prefix+self.interpfefile,np.vstack((xfine,yi_t,yider_t)).T)
            if not _dont_save_hist:
                np.savetxt(self.prefix+self.histfile, np.vstack((xfa,pf)).T)


    def set_harmonic_u_corr(self,K=0.):
        if self.corrs is None:
            raise Exception("Please calculate correlation functions first.")
        if K==0.:
            self.ucorr=pd.DataFrame({"au": np.zeros(len(self.corrs.index)), "vu": np.zeros(len(self.corrs.index))}, index=self.corrs.index)
        else:
            if self.first_order:
                raise Exception("Harmonic first order not implemented (for K!=0).")
            else:
                self.ucorr=pd.DataFrame({"au": -K*self.corrs["vv"]},index=self.corrs.index)

    def compute_au_corr(self, *args, **kwargs):
        print("WARNING: This function has been renamed to compute_u_corr, please change.")
        self.compute_u_corr(*args, **kwargs)

    def compute_u_corr(self):
        if self.fe_spline is None:
            raise Exception("Free energy has not been computed.")
        if self.verbose:
            print("Calculate a/v grad(U(x)) correlation function...")

        # get target length from first element and trunc
        ncorr=self.xva_list[0][self.xva_list[0].index < self.trunc].shape[0]


        self.ucorr=pd.DataFrame({"au":np.zeros(ncorr)}, \
        index=self.xva_list[0][self.xva_list[0].index < self.trunc].index\
              -self.xva_list[0].index[0])
        if self.first_order:
            self.ucorr["vu"]=np.zeros(ncorr)

        for weight,xva in zip(self.weights,self.xva_list):
            x=xva["x"].values
            a=xva["a"].values
            corr=correlation(a,self.dU(x),subtract_mean=False)
            self.ucorr["au"]+=weight*corr[:ncorr]

            if self.first_order:
                v=xva["v"].values
                corr=correlation(v,self.dU(x),subtract_mean=False)
                self.ucorr["vu"]+=weight*corr[:ncorr]

        self.ucorr/=self.weightsum

        if self.saveall:
            self.ucorr.to_csv(self.prefix+self.ucorrfile,sep=" ")

    def compute_corrs(self):
        if self.verbose:
            print("Calculate vv, va and aa correlation functions...")

        self.corrs=None
        for weight,xva in zip(self.weights,self.xva_list):
            vvcorrw=weight*pdcorr(xva,"v","v",self.trunc,"vv")
            vacorrw=weight*pdcorr(xva,"v","a",self.trunc,"va")
            aacorrw=weight*pdcorr(xva,"a","a",self.trunc,"aa")
            if self.corrs is None:
                self.corrs=pd.concat([vvcorrw,vacorrw,aacorrw],axis=1)
            else:
                self.corrs["vv"]+=vvcorrw["vv"]
                self.corrs["va"]+=vacorrw["va"]
                self.corrs["aa"]+=aacorrw["aa"]
        #print(self.corrs)
        self.corrs/=self.weightsum
        #print(self.corrs)

        if self.saveall:
            self.corrs.to_csv(self.prefix+self.corrsfile,sep=" ")

    def compute_kernel(self, first_order=None, k0=0.):
        """
Computes the memory kernel. If you give a nonzero value for k0, this is used at time zero, if set to 0, the C-routine will calculate k0 from the second order memory equation.
        """
        if first_order is None:
            first_order=self.first_order
        if first_order and not self.first_order:
            raise Excpetion("Please initialize in first order mode, which allows both first and second order.")
        if self.corrs is None or self.ucorr is None:
            raise Exception("Need correlation functions to compute the kernel.")
        if self.mass is None:
            if self.verbose:
                print("Mass not calculated.")
            self.compute_mass()

        v_acf=self.corrs["vv"].values
        va_cf=self.corrs["va"].values
        dt=self.corrs.index[1]-self.corrs.index[0]

        if first_order:
            vu_cf=self.ucorr["vu"].values
        #else: #at the moment
        a_acf=self.corrs["aa"].values
        au_cf=self.ucorr["au"].values

        if self.verbose:
            print("Use dt:",dt)

        kernel=np.zeros(len(v_acf))

        if first_order:
            ckernel.ckernel_first_order_core(v_acf,va_cf*self.mass,a_acf*self.mass,vu_cf,au_cf,dt,k0,kernel)
        else:
            ckernel.ckernel_core(v_acf,va_cf,a_acf*self.mass,au_cf,dt,k0,kernel)


        ikernel=cumtrapz(kernel,dx=dt,initial=0.)

        self.kernel=pd.DataFrame({"k":kernel,"ik":ikernel},index=self.corrs.index)
        self.kernel=self.kernel[["k","ik"]]
        if self.saveall:
            if first_order:
                self.kernel.to_csv(self.prefix+self.kernelfile_1st,sep=" ")
            else:
                self.kernel.to_csv(self.prefix+self.kernelfile,sep=" ")

        return self.kernel

    def dU(self,x):
        if self.fe_spline is None:
            raise Exception("Free energy has not been computed.")
        if self.per:
            if self.x0_fe is None or self.x1_fe is None:
                raise Exception("Please compute free energy after setting p.b.c.")
            assert(self.x1_fe-self.x0_fe==self.x1-self.x0)
            yi = interpolate.splev((x-self.x0_fe)%(self.x1_fe-self.x0_fe)+self.x0_fe, self.fe_spline, der=1,ext=2)*self.kT
        else:
            yi = interpolate.splev(x, self.fe_spline, der=1)*self.kT
        return yi

    def load(self, prefix=None):
        if prefix is None:
            prefix=self.prefix

        if os.path.isfile(prefix+self.corrsfile):
            print("Found correlation functions.")
            self.corrs=pd.read_csv(prefix+self.corrsfile, sep=" ", index_col=0)
            self.dt=self.corrs.index[1]-self.corrs.index[0]

        if os.path.isfile(prefix+self.histfile):
            print("Found free energy histogram.")
            lhist=np.loadtxt(prefix+self.histfile)
            fehist=[lhist[:,1].ravel(),lhist[:,0].ravel()]
            print("Interpolate...")
            self.compute_fe(fehist=fehist,_dont_save_hist=True)

        if os.path.isfile(prefix+self.ucorrfile):
            print("Found potential correlation functions.")
            self.ucorr=pd.read_csv(prefix+self.ucorrfile, sep=" ", index_col=0)
            self.dt=self.ucorr.index[1]-self.ucorr.index[0]

        if os.path.isfile(prefix+self.kernelfile):
            print("Found second kind kernel.")
            self.kernel=pd.read_csv(prefix+self.kernelfile, sep=" ", index_col=0)
            self.dt=self.kernel.index[1]-self.kernel.index[0]

        if os.path.isfile(prefix+self.kernelfile_1st):
            print("Found first kind kernel.")
            self.kernel_1st=pd.read_csv(prefix+self.kernelfile_1st, sep=" ", index_col=0)
            self.dt=self.kernel_1st.index[1]-self.kernel_1st.index[0]

        print("Found dt =", self.dt)
