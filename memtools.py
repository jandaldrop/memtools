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

    spdiffs=sdiffs.shift(-1)+sdiffs

    xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":sdiffs["x"]/(2*dt),"a":ddiffs["x"]/(dt**2)},index=xf.index)
    #xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":sdiffs["x"]/(2*dt),"a":spdiffs["x"]/(4*dt**2)},index=xf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

def compute_a_forward(xvf):
    diffs=xvf-xvf.shift(1)
    dt=xvf.iloc[1]["t"]-xvf.iloc[0]["t"]

    xva=pd.DataFrame({"t":xvf["t"],"x":xvf["x"],"v":xvf["v"],"a":diffs["v"]/(dt)},index=xvf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()


def compute_va_forward(xf, correct_jumps=False, jump=360, jump_thr=270):
    diffs=xf-xf.shift(1)
    dt=xf.iloc[1]["t"]-xf.iloc[0]["t"]
    if correct_jumps:
        diffs.loc[diffs["x"] < jump_thr,"x"]+=jump
        diffs.loc[diffs["x"] > jump_thr,"x"]-=jump

    ddiffs=diffs.shift(-1)-diffs
    #sdiffs=diffs.shift(-1)+diffs

    #spdiffs=sdiffs.shift(-1)+sdiffs

    xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":diffs["x"]/(dt),"a":ddiffs["x"]/(dt**2)},index=xf.index)
    #xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":sdiffs["x"]/(2*dt),"a":spdiffs["x"]/(4*dt**2)},index=xf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()

def compute_a_clean(xvf):
    xvf1=xvf.iloc[1::2]
    xvf2=xvf.iloc[::2]
    diffs=xvf2-xvf2.shift(1)
    dt=xvf.iloc[1]["t"]-xvf.iloc[0]["t"]

    xva=pd.DataFrame({"t":xvf1["t"],"x":xvf1["x"],"v":xvf1["v"],"a":diffs["v"]/(dt)},index=xvf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()


