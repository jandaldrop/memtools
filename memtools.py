from __future__ import print_function
import numpy as np
import pandas as pd
from igle import *
from igleplot import *
from correlation import *
from flist import *

def ver():
    print("This is memtools version 1.06")


def xframe(x, time, round_time=1.e-4, fix_time=True, dt=-1):
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
    diffs=xvf.shift(-1)-xvf.shift(1)
    dt=xvf.iloc[1]["t"]-xvf.iloc[0]["t"]
    xva=pd.DataFrame({"t":xvf["t"],"x":xvf["x"],"v":xvf["v"],"a":diffs["v"]/(2.*dt)},index=xvf.index)
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

    xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":sdiffs["x"]/(2.*dt),"a":ddiffs["x"]/dt**2},index=xf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()
