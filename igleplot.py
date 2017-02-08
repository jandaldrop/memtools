from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from correlation import *
from memtools import *
from igle import *

class IglePlot(Igle):
    def __init__(self, *args, **kwargs):
        super(IglePlot, self).__init__(*args, **kwargs)
        self.plot=True

    def compute_corrs(self, *args, **kwargs):
        ret=super(IglePlot, self).compute_corrs(*args, **kwargs)
        if self.plot:
            self.plot_corrs()
        return ret

    def compute_au_corr(self, *args, **kwargs):
        ret=super(IglePlot, self).compute_au_corr(*args, **kwargs)
        if self.plot:
            self.plot_au_corr()
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


    def plot_au_corr(self):
        plt.figure()
        plt.plot(self.aucorr.index,self.aucorr["au"])
        plt.xscale("log")
        plt.xlabel("$t$")
        plt.ylabel("$\\langle a\\nabla U\\rangle$")
        plt.title("$\\langle a\\nabla U\\rangle$ correlation function")
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
