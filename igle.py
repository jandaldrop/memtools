from __future__ import print_function
import numpy as np
import pandas as pd
from correlation import *
from memtools import *
from scipy import interpolate
from scipy.integrate import cumtrapz
import ckernel

class Igle(object):
    def __init__(self,xva_arg,saveall=True,prefix="",verbose=True,kT=2.494,trunc=1.,__override_time_check__=False):
        """ xva_arg should be either a pandas timeseries or an iterable collection (i.e. list) of them. """
        if isinstance(xva_arg,pd.DataFrame):
            self.xva_list=[xva_arg]
        else:
            self.xva_list=xva_arg
        for xva in self.xva_list:
            for col in ['t','x','v','a']:
                if col not in xva.columns:
                    raise Exception("Please provide txva data frame, or an iterable collcetion (i.e. list) of txva data frames. And not some other shit.")
        self.saveall=saveall
        self.prefix=prefix
        self.verbose=verbose
        self.kT=kT

        # filenames
        self.corrsfile="corrs.txt"
        self.interpfefile="interp-fe.txt"
        self.histfile="fe-hist.txt"
        self.aucorrfile="au-corr.txt"
        self.kernelfile="kernel.txt"

        self.corrs=None
        self.aucorr=None
        self.mass=None
        self.fe_spline=None
        self.fe=None
        self.per=False

        # processing input arguments
        self.weights=np.array([xva.shape[0] for xva in self.xva_list],dtype=int)
        self.weightsum=np.sum(self.weights)

        if self.verbose:
            print("Found trajectories with the following legths:")
            print(self.weights)

        lastinds=np.array([xva.index[-1] for xva in self.xva_list],dtype=int)
        smallest=np.min(lastinds)
        if smallest < trunc:
            if self.verbose:
                print("Warning: Found a trajectory shorter than the argument trunc. Override.")
            trunc=smallest
        self.trunc=trunc


        if not __override_time_check__:
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

        v2sum=0.
        for i,xva in enumerate(self.xva_list):
            v2sum+=(xva["v"]**2).mean()*self.weights[i]
        v2=v2sum/self.weightsum
        self.mass=self.kT/v2

        if self.verbose:
            print("Found mass:", self.mass)

    def compute_fe(self,bins="auto",fehist=None):
        '''Computes the free energy. If you run into memory problems, you can provide an histogram.'''
        if self.verbose:
            print ("Calculate histogram...")
        if fehist is None:
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

        self.fe_spline= interpolate.splrep(xf, fe, s=0, per=self.per)
        self.fe=pd.DataFrame({"F":fe},index=xf)

        if self.saveall:
            dxf=xf[1]-xf[0]
            xfine=np.arange(xf[0],xf[-1],dxf/10.)
            yi_t=interpolate.splev(xfine, self.fe_spline)
            yider_t=interpolate.splev(xfine, self.fe_spline, der=1)
            np.savetxt(self.prefix+self.interpfefile,np.vstack((xfine,yi_t,yider_t)).T)
            np.savetxt(self.prefix+self.histfile, np.vstack((xfa,pf)).T)


    def set_harmonic_au_corr(self,K=0.):
        if self.corrs is None:
            raise Exception("Please calculate correlation functions first.")
        self.aucorr=pd.DataFrame({"au": -K*self.corrs["vv"]},index=self.corrs.index)


    def compute_au_corr(self):
        if self.fe_spline is None:
            raise Exception("Free energy has not been computed.")
        if self.verbose:
            print("Calculate a grad(U(x)) correlation function...")

        # get target length from first element and trunc
        ncorr=self.xva_list[0][self.xva_list[0].index < self.trunc].shape[0]
        self.aucorr=pd.DataFrame({"au":np.zeros(ncorr)}, index=self.xva_list[0][self.xva_list[0].index < self.trunc].index)

        for weight,xva in zip(self.weights,self.xva_list):
            x=xva["x"].values
            a=xva["a"].values
            corr=correlation(a,self.dU(x),subtract_mean=False)
            self.aucorr["au"]+=weight*corr[:ncorr]

        self.aucorr["au"]/=self.weightsum

        if self.saveall:
            self.aucorr.to_csv(self.prefix+self.aucorrfile,sep=" ")

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

    def compute_kernel(self, use_c=True):
        if self.corrs is None or self.aucorr is None:
            raise Exception("Need correlation functions to compute the kernel.")
        if self.mass is None:
            if self.verbose:
                print("Mass not calculated.")
            self.compute_mass()

        v_acf=self.corrs["vv"].values
        va_cf=self.corrs["va"].values
        a_acf=self.corrs["aa"].values
        au_cf=self.aucorr["au"].values
        dt=self.xva_list[0].index[1]-self.xva_list[0].index[0]

        if self.verbose:
            print("Use dt:",dt)

        kernel=np.zeros(len(v_acf))

        #use the c++ routine?
        if use_c:
            ckernel.ckernel_core(v_acf,va_cf,a_acf*self.mass,au_cf,dt,kernel)
        #otherwise use python implementation
        else:
            def w(j,i):
                if j==0 or j==i:
                    return 0.5
                else:
                    return 1.

            k0=(self.mass*a_acf[0]+au_cf[0])/v_acf[0]
            if self.verbose:
                print("k0="+str(k0)+"\n")
            kernel[0]=k0
            prefac=1./(v_acf[0]+va_cf[0]*dt*w(0,0))

            for i in range(1,len(kernel)):
                num=self.mass*a_acf[i]+au_cf[i]
                for j in range(0,i):
                    num-=dt*w(j,i)*va_cf[i-j]*kernel[j]

                kernel[i]=prefac*num

        # --- end of calculation ---

        ikernel=cumtrapz(kernel,dx=dt,initial=0.)

        self.kernel=pd.DataFrame({"k":kernel,"ik":ikernel},index=self.corrs.index)
        self.kernel=self.kernel[["k","ik"]]
        if self.saveall:
            self.kernel.to_csv(self.prefix+self.kernelfile,sep=" ")

        return self.kernel

    def dU(self,x):
        if self.fe_spline is None:
            raise Exception("Free energy has not been computed.")
        if self.per:
            yi = interpolate.splev(x%(self.x1-self.x0)+self.x0, self.fe_spline, der=1,ext=2)*self.kT
        else:
            yi = interpolate.splev(x, self.fe_spline, der=1,ext=2)*self.kT
        return yi
