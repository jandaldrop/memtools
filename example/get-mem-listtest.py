from memtools import *
import numpy as np
import pandas as pd
import mdtraj

#memtools.ver()


traj=mdtraj.load("prd.xtc",top="alkane.gro")
dih=np.rad2deg(mdtraj.compute_dihedrals(traj, [[0,1,2,3]], periodic=True, opt=True))

#q1=int(dih.shape[0]/3)
#q2=2*int(dih.shape[0]/3)

q1=10000
q2=20000

dih1=dih[:q1,:]
dih2=dih[q1:q2,:]
dih3=dih[q2:,:]

time=traj.time
t1=time[:q1]
t2=time[q1:q2]
t3=time[:q1]


xf1=xframe(dih1,t1)
xf2=xframe(dih2,t2)
xf3=xframe(dih3,t3)

xva1=compute_va(xf1,correct_jumps=True)
xva2=compute_va(xf2,correct_jumps=True)
xva3=compute_va(xf3,correct_jumps=True)

xva_list=[xva1,xva2,xva3]

mymem=Igle(xva_list,trunc=10)
mymem.compute_corrs()
mymem.set_periodic()
mymem.compute_fe()
mymem.compute_au_corr()

mymem.compute_kernel()
