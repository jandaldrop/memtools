from igleplot import *
import numpy as np


class IgleC(IglePlot):

  def __init__(self, *args, **kwargs):
    self.corrs_index=0
    self.kwargs = kwargs
    super(IgleC, self).__init__(*args, **kwargs)

  def push_back_xva(self,xva_arg):
    super(IgleC, self).__init__(xva_arg,self.kwargs)
    self.compute_corrs(self)

  def compute_corrs(self):
    self.self.corrsfile="corrs_"+str(self.corrs_index)+".txt"
    super(IgleC,self).compute_corrs()
    np.savetxt(self.weightsum,"weights_"+str(self.corrs_index))
    self.corrs_index=self.corrs_index+1

  def compute_kernel(self, **kwargs):
    for i in range(self.corrs_index):
        weight=np.loadtxt("weights_"+str(self.corrs_index))
        if corr is None:
            corr=weight*pd.read_csv(self.prefix+"corrs_"+str(self.corrs_index)+".txt", sep=" ", index_col=0)
        corr+=weight*pd.read_csv(self.prefix+"corrs_"+str(self.corrs_index)+".txt", sep=" ", index_col=0)
        totWeight+=weight
    self.corrs=corr/totWeight
    super(IgleC,self).compute_kernel()
