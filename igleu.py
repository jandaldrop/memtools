from igleplot import *
from enum import Enum
import numpy as np

class Potentials(Enum):
    HARMONIC = 1
    DOUBLEWELL = 2


class IgleU(IglePlot):

  def __init__(self, *args, **kwargs):
    super(IgleU, self).__init__(*args, **kwargs)
    self.potential = Potentials.HARMONIC
    self.fe_spline = True

  def dU(self,x):
    if(self.potential==Potentials.HARMONIC):
      return 2*x*self.kT
    elif(self.potential==Potentials.DOUBLEWELL):
      return 4*np.power(x,3)-4*x*self.kT
    else:
      print("WARNING: analytic potential type not implemented")
      return 0.0


