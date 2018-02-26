import numpy as np
import pandas as pd

def ver():
    print("This is memtools version 1.1")


def xframe(x, time, round_time=1.e-4, fix_time=True, dt=-1):
    """
    is saved twice, as an index and as a separate field.

    Parameters
    ----------
    x : numpy array
        The time series.
    time : numpy array
        The respective time values.
    round_time : float, default=1.e-4
        When fix_time is set times are rounded to this value.
    fix_time : bool, default=True
        Round first timestep to round_time precision and replace times.
    dt : float, default=-1
        When positive, this value is used for fixing the time instead of
        the first timestep.
    """
    x=np.asarray(x)
    time=np.asarray(time)
    if fix_time:
        if dt<0:
            dt=np.round((time[1]-time[0])/round_time)*round_time
        time=np.linspace(0.,dt*x.size,x.size)
    df = pd.DataFrame({"t":time.ravel(),"x":x.ravel()}, index=time.ravel())
    df.index.name='#t'
    return df

def compute_a(xvf):
    """

    Parameters
    ----------
    xvf : pandas dataframe (['t', 'x', 'v'])
    """
    diffs=xvf.shift(-1)-xvf.shift(1)
    dt=xvf.iloc[1]["t"]-xvf.iloc[0]["t"]
    xva=pd.DataFrame({"t":xvf["t"],"x":xvf["x"],"v":xvf["v"],"a":diffs["v"]/(2.*dt)},index=xvf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

def compute_va(xf, correct_jumps=False, jump=360, jump_thr=270):
    """
    returned by xframe.

    Parameters
    ----------
    xf : pandas dataframe (['t', 'x'])

    correct_jumps : bool, default=False
        Jumps in the trajectory are removed (relevant for periodic data).
    jump : float, default=360
        The size of a jump.
    jump_thr : float, default=270
        Threshold for jump detection.
    """
    diffs=xf-xf.shift(1)
    dt=xf.iloc[1]["t"]-xf.iloc[0]["t"]
    if correct_jumps:
        diffs.loc[diffs["x"] < jump_thr,"x"]+=jump
        diffs.loc[diffs["x"] > jump_thr,"x"]-=jump

    ddiffs=diffs.shift(-1)-diffs
    sdiffs=diffs.shift(-1)+diffs

    xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":sdiffs["x"]/(2.*dt),"a":ddiffs["x"]/dt**2},index=xf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()
import numpy as np
import pandas as pd
import os.path
#import ckernel


import sys
if sys.version_info < (3,0):
    print("Import python3 like zip function.")

class Igle(object):
    """
    The main class for the memory extraction, holding all data and the extracted memory kernels.
    """
    def __init__(self,xva_arg,saveall=True,prefix="",verbose=True,kT=2.494,trunc=1.,__override_time_check__=False,initial_checks=True,first_order=True,corrs_from_der=False):
        """
        Create an instance of the Igle class.

        Parameters
        ----------
        xva_arg : pandas dataframe ()['t', 'x', 'v', 'a']) or list of dataframes.
            Use compute_va() or see its output for format details.
            The timeseries to analyze. It should be either a pandas timeseries
            or a listlike collection of them. Set xva_arg=None for load mode.
        saveall : bool, default=True
            Whether to save all output functions.
        prefix : str
            Prefix for the saved output functions.
        verbose : bool, default=True
            Set verbosity.
        kT : float, default=2.494
            Numerical value for kT.
        trunc : float, default=1.0
            Truncate all correlation functions and the memory kernel after this
            time value.
        __override_time_check__ : bool, default=False
            Override initial time check when a list of trajectories is provided.
        initial_checks : bool, default=True
            Do a few initial consistency checks.
        first_order : bool, default=True
            Use a Volterra equation of the first kind. If this option is set to
            True, the Volterra equation of the second kind can be selected in
            the compute_kernel function. If it is set to False, only the Volerra
            equation of the second kind is used.
        corrs_from_der : bool, default=False
            (Do not use.)
        """
        if xva_arg is not None:
            if isinstance(xva_arg,pd.DataFrame):
                self.xva_list=[xva_arg]
            else:
                self.xva_list=xva_arg
            if isinstance(self.xva_list,flist) and initial_checks:
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
        self.corrs_from_der=corrs_from_der
        if self.corrs_from_der:
            print("WARNING: corrs_from_der=True is not properly tested.")

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
        """
        Set periodic boundary conditions.

        Parameters
        ----------
        x0 : float, default=-180
            Lower interval bound.

        x1 : float, default=180
            Upper interval bound.
        """
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
        '''
        interpolation. You can alternatively provide an histogram.

        Parameters
        ----------

        bins : str, or int, default="auto"
            The number of bins. It is passed to the numpy.histogram routine,
            see its documentation for details.
        fehist : list, default=None
            Provide a (precomputed) histogram in the format as returned by
            numpy.histogram.
        _dont_save_hist : bool, default=False
            Do not save the histogram.
        '''
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
        """
        Set an harmonic potential (instead of calculating the free energy.)

        Parameters
        ----------

        K : float, default=0
            Potential strength.
        """
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

    def compute_u_corr(self, edge_order=2):
        """
        Compute the correlation function(s) including the potential.

        Parameters
        ----------

        edge_order : int, default=2
            egde_order used by numpy.gradient (only relevant for corrs_from_der=True).
        """
        if self.fe_spline is None:
            raise Exception("Free energy has not been computed.")
        if self.verbose:
            print("Calculate a/v grad(U(x)) correlation function...")

        ncorr=self.xva_list[0][self.xva_list[0].index < self.trunc].shape[0]

        if self.corrs_from_der:
            self.ucorr=pd.DataFrame({"xu":np.zeros(ncorr)}, \
                index=self.xva_list[0][self.xva_list[0].index < self.trunc].index\
                -self.xva_list[0].index[0])
        else:
            self.ucorr=pd.DataFrame({"au":np.zeros(ncorr)}, \
                index=self.xva_list[0][self.xva_list[0].index < self.trunc].index\
                -self.xva_list[0].index[0])

        if self.first_order:
            self.ucorr["vu"]=np.zeros(ncorr)

        for weight,xva in zip(self.weights,self.xva_list):
            x=xva["x"].values
            if self.corrs_from_der:
                corr=correlation(x,self.dU(x),subtract_mean=False)
                self.ucorr["xu"]+=weight*corr[:ncorr]
            else:
                a=xva["a"].values
                corr=correlation(a,self.dU(x),subtract_mean=False)
                self.ucorr["au"]+=weight*corr[:ncorr]

                if self.first_order:
                    v=xva["v"].values
                    corr=correlation(v,self.dU(x),subtract_mean=False)
                    self.ucorr["vu"]+=weight*corr[:ncorr]

        self.ucorr/=self.weightsum
        if self.corrs_from_der:
            dt=self.ucorr.index[1]-self.ucorr.index[0]
            self.ucorr["vu"]=-np.gradient(self.ucorr["xu"].values, dt, edge_order=edge_order)
            self.ucorr["au"]=-np.gradient(self.ucorr["vu"].values, dt, edge_order=edge_order)

        if self.saveall:
            self.ucorr.to_csv(self.prefix+self.ucorrfile,sep=" ")

    def compute_corrs(self, edge_order=2):
        """
        Compute correlation functions without the potential.

        Parameters
        ----------

        edge_order : int, default=2
            egde_order used by numpy.gradient (only relevant for corrs_from_der=True).
        """

        if self.verbose:
            print("Calculate vv, va and aa correlation functions...")

        self.corrs=None
        if self.corrs_from_der:
            for weight,xva in zip(self.weights,self.xva_list):
                xxcorrw=weight*pdcorr(xva,"x","x",self.trunc,"xx")
                if self.corrs is None:
                    self.corrs=xxcorrw
                else:
                    self.corrs["xx"]+=xxcorrw["xx"]
            self.corrs/=self.weightsum

            dt=self.corrs.index[1]-self.corrs.index[0]
            self.corrs["xv"]=np.gradient(self.corrs["xx"].values, dt, edge_order=edge_order)
            self.corrs["vv"]=-np.gradient(self.corrs["xv"].values, dt, edge_order=edge_order)
            self.corrs["va"]=np.gradient(self.corrs["vv"].values, dt, edge_order=edge_order)
            self.corrs["aa"]=-np.gradient(self.corrs["va"].values, dt, edge_order=edge_order)

        else:
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
        Computes the memory kernel.

        Parameters
        ----------
        first_order : bool, default=None
            Choose whether the Volterra equation of the first kind (and not of
            the second kind) is used. Only works when first_order=True was set
            on initialization.

        k0 : float, default=0.
            If you give a nonzero value for k0, this is used at time zero, if set to 0,
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
        """

        Parameters
        ----------

        prefix : str, default=None
            Prefix for the filenames.
        """
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class IglePlot(Igle):
    """
    It is very useful for explorative analysis in jupyter notebooks.
    All arguments are passed to Igle, so for documentation and Tab completion
    please refer to the Igle class.
    """
    def __init__(self, *args, **kwargs):
        super(IglePlot, self).__init__(*args, **kwargs)
        self.plot=True

    def compute_corrs(self, *args, **kwargs):
        ret=super(IglePlot, self).compute_corrs(*args, **kwargs)
        if self.plot:
            self.plot_corrs()
        return ret

    def compute_au_corr(self, *args, **kwargs):
        print("WARNING: This function has been renamed to compute_u_corr, please change.")
        self.compute_u_corr(*args, **kwargs)

    def compute_u_corr(self, *args, **kwargs):
        ret=super(IglePlot, self).compute_u_corr(*args, **kwargs)
        if self.plot:
            self.plot_u_corr()
        return ret

    def compute_fe(self, *args, **kwargs):
        ret=super(IglePlot, self).compute_fe(*args, **kwargs)
        if self.plot:
            self.plot_fe()
        return ret

    def compute_kernel(self, *args, **kwargs):
        ret=super(IglePlot, self).compute_kernel(*args, **kwargs)
        if self.plot:
            self.plot_kernel()
        return ret

    def plot_corrs(self):
        plt.figure()
        plt.plot(self.corrs.index,self.corrs["vv"])
        plt.xscale("log")
        plt.xlabel("$t$")
        plt.ylabel("$\\langle vv\\rangle$")
        plt.title("Velocity autocorrelation function")
        plt.show(block=False)

        plt.figure()
        plt.plot(self.corrs.index,self.corrs["va"])
        plt.xscale("log")
        plt.xlabel("$t$")
        plt.ylabel("$\\langle va\\rangle$")
        plt.title("Velocity acceleration correlation function")
        plt.show(block=False)

        plt.figure()
        plt.plot(self.corrs.index,self.corrs["aa"])
        plt.xscale("log")
        plt.xlabel("$t$")
        plt.ylabel("$\\langle aa\\rangle$")
        plt.title("Acceleration autocorrelation function")
        plt.show(block=False)


    def plot_u_corr(self):
        plt.figure()
        plt.plot(self.ucorr.index,self.ucorr["au"])
        plt.xscale("log")
        plt.xlabel("$t$")
        plt.ylabel("$\\langle a\\nabla U\\rangle$")
        plt.title("$\\langle a\\nabla U\\rangle$ correlation function")
        plt.show(block=False)
        if self.first_order:
            plt.figure()
            plt.plot(self.ucorr.index,self.ucorr["vu"])
            plt.xscale("log")
            plt.xlabel("$t$")
            plt.ylabel("$\\langle v\\nabla U\\rangle$")
            plt.title("$\\langle v\\nabla U\\rangle$ correlation function")
            plt.show(block=False)

    def plot_fe(self, nxfine=1000):
        plt.figure()
        plt.plot(self.fe.index,self.fe["F"],"o")

        x0=self.fe.index.values[0]
        x1=self.fe.index.values[-1]
        xfine=np.arange(x0,x1,(x1-x0)/nxfine)

        plt.plot(xfine,interpolate.splev(xfine, self.fe_spline))

        plt.xlabel("$x$")
        plt.ylabel("$F$ [kT]")
        plt.title("Free energy")
        plt.show(block=False)

    def plot_kernel(self):
        plt.figure()
        plt.plot(self.kernel.index,self.kernel["k"])
        plt.xscale("log")
        plt.xlabel("$t$")
        plt.ylabel("$\\Gamma$")
        plt.title("Memory kernel")
        plt.show(block=False)
class flist(list):
    """
Listlike object, where the getitem operation is
replaced by a function call.
I.e. flist can store a list of filenames, which
are loaded into memory only on access.
The result of the last function call is stored.
The number of function calls are counted in flist.n_calls
    """
    def __init__(self, f_handle, r=list()):
        super(flist, self).__init__(r)
        self.f_handle=f_handle
        self.n_calls=0
        self.last_arg=None
    def __iter__(self):
        for x in list.__iter__(self):
            yield self.f(x)
    def __getitem__(self, n):
        return self.f(super(flist, self).__getitem__(n))
    def f(self, x):
        if self.last_arg!=x:
            self.last_res=self.f_handle(x)
            self.last_arg=x
            self.n_calls+=1
        return self.last_res
