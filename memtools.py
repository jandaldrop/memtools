from __future__ import print_function
import numpy as np
import pandas as pd
import mdtraj
from igle import *
from igleplot import *
from igleu import *
from correlation import *


def ver():
    print("This is memtools version 0.61")


def xframe(x,time,round_time=1.e-4,fix_time=True,dt=-1):
    x=np.asarray(x)
    time=np.asarray(time)
    if fix_time:
        if dt<0:
            dt=np.round((time[1]-time[0])/round_time)*round_time
        time=np.arange(0.,dt*x.size,dt)
    df = pd.DataFrame({"t":time.flatten(),"x":x.flatten()}, index=np.round(time/round_time)*round_time)
    df.index.name='#t'
    return df

def compute_va(xf, correct_jumps=False, jump=360, jump_thr=270):
    diffs=xf-xf.shift(1)
    if correct_jumps:
        diffs.loc[diffs["x"] < jump_thr,"x"]+=jump
        diffs.loc[diffs["x"] > jump_thr,"x"]-=jump

    ddiffs=diffs.shift(-1)-diffs

    xva=pd.DataFrame({"t":xf["t"],"x":xf["x"],"v":diffs["x"]/diffs["t"],"a":ddiffs["x"]/diffs["t"]**2},index=xf.index)
    xva = xva[['t', 'x', 'v', 'a']]
    xva.index.name='#t'

    return xva.dropna()
