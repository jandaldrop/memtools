import numpy as np
import pandas as pd

def correlation(a,b=None,subtract_mean=False):
    meana = int(subtract_mean)*np.mean(a)
    a2 = np.append(a-meana,
np.zeros(2**int(np.ceil((np.log(len(a))/np.log(2))))-len(a)))
    data_a = np.append(a2, np.zeros(len(a2)))
    fra = np.fft.fft(data_a)
    if b is None:
        sf = np.conj(fra)*fra
    else:
        meanb = int(subtract_mean)*np.mean(b)
        b2 = np.append(b-meanb,
np.zeros(2**int(np.ceil((np.log(len(b))/np.log(2))))-len(b)))
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft(data_b)
        sf = np.conj(fra)*frb
    res = np.fft.ifft(sf)
    cor = np.real(res[:len(a)])/np.array(range(len(a),0,-1))
    return cor

def pdcorr(df,f1,f2,trunc=None,oname="c"):
    a=df.loc[:,f1].values
    if f1 == f2:
        corr=correlation(a)
    else:
        b=df.loc[:,f2].values
        corr=correlation(a,b)

    cf=pd.DataFrame({oname:corr}, index=df.index-df.index[0])

    if not trunc is None:
        cf=cf[df.index<trunc]

    return cf
