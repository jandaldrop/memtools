from __future__ import print_function
import numpy as np
import pandas as pd
from igle import *
from igleplot import *
from igleu import *
from iglec import *
from correlation import *


def ver():
    print("This is memtools version 0.90")


def xframe(x, time, round_time=1.e-4, fix_time=True, dt=-1):
    x=np.asarray(x)
    time=np.asarray(time)
    if fix_time:
        if dt<0:
            dt=np.round((time[1]-time[0])/round_time)*round_time
        time=np.arange(0.,dt*x.size,dt)
    df = pd.DataFrame({"t":time.flatten(),"x":x.flatten()}, index=np.round(time/round_time)*round_time)
    df.index.name='#t'
    return df

def xvframe(x,v,time,round_time=1.e-4,fix_time=True,dt=-1):
    x=np.asarray(x)
    v=np.asarray(v)
    time=np.asarray(time)
    if fix_time:
        if dt<0:
            dt=np.round((time[1]-time[0])/round_time)*round_time
        time=np.arange(0.,dt*x.size,dt)
    df = pd.DataFrame({"t":time.flatten(),"x":x.flatten(),"v":v.flatten()}, index=np.round(time/round_time)*round_time)
    df.index.name='#t'
    return df

def compute_a(xvf):
    diffs=xvf-xvf.shift(1)
    dt=xvf.iloc[1]["t"]-xvf.iloc[0]["t"]

    sdiffs=diffs.shift(-1)+diffs
    xva=pd.DataFrame({"t":xvf["t"],"x":xvf["x"],"v":xvf["v"],"a":sdiffs["v"]/(2.*dt)},index=xvf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

def compute_va(xf, correct_jumps=False, jump=360, jump_thr=270):
    diffs=xf-xf.shift(1)
    dt=xf.iloc[1]["t"]-xf.iloc[0]["t"]
    if correct_jumps:
        diffs.loc[diffs["x"] < jump_thr,"x"]+=jump
        diffs.loc[diffs["x"] > jump_thr,"x"]-=jump

    ddiffs=diffs.shift(-1)-diffs
    sdiffs=diffs.shift(-1)+diffs

    xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":sdiffs["x"]/(2*dt),"a":ddiffs["x"]/(dt**2)},index=xf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

# if velocities are known from the simulation this routine will compute acceleration only by the forward averages
# the va correlation function is then decaying in one timestep not smear out the delta-function in the random force
def compute_a_delta(xvf):
    diffs=xvf-xvf.shift(1)
    dt=xvf.iloc[1]["t"]-xvf.iloc[0]["t"]

    xva=pd.DataFrame({"t":xvf["t"],"x":xvf["x"],"v":xvf["v"],"a":diffs["v"]/(dt)},index=xvf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

def computeFrozenKernel(numpy_tf,trunc=None, file=None,kT=2.4943,m=1.):
    t=numpy_tf[:,0]
    corr = correlation(numpy_tf[:,1],subtract_mean=True)/(m*kT)
    ikernel=cumtrapz(corr,x=t,initial=0.)
    kernel_ff=pd.DataFrame({"k":corr,"ik":ikernel},index=t)
    if not trunc is None:
        kernel_ff=kernel_ff[kernel_ff.index<trunc]
    if file is not None:
        kernel_ff.to_csv(file,sep=" ")
    return kernel_ff
