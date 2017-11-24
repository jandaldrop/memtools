from memtools import *
import numpy as np
import pandas as pd
import mdtraj

#memtools.ver()


traj=mdtraj.load("../newanalysis-dev-singletrj/prd.xtc",top="alkane.gro")
dih=np.rad2deg(mdtraj.compute_dihedrals(traj, [[0,1,2,3]], periodic=True, opt=True))

xf=xframe(dih,traj.time)
xvaf=compute_va(xf,correct_jumps=True)
#print xvaf

mymem=IglePlot(xvaf,trunc=10)
#mymem=Igle(xvaf,trunc=10)

mymem.compute_corrs()
mymem.set_periodic()
mymem.compute_fe()
mymem.compute_u_corr()

mymem.compute_kernel()
