import nitime.algorithms as tsa
import numpy as np
from matplotlib import pyplot as plt

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)


def nextpower2(x):
    return 1 << (x - 1).bit_length()


def getgrid(Fs, nfft):
    df = Fs / float(nfft)
    f = np.arange(0, Fs, df)
    f = f[0:nfft]
    fpass = Fs / 2
    f = f[f <= fpass]
    return f


def mtspecgram(sig, Fs, BW, window, timestep, plot = False, ax=None):
    Nwin = int(round(window * Fs))
    Nstep = int(round(timestep * Fs))
    N = len(sig)
    winstart = np.arange(0, N - Nwin + 1, Nstep)
    nw = len(winstart)

    nfft = max(nextpower2(Nwin), Nwin)
    f = getgrid(Fs, nfft)
    Nf = len(f)
    S = np.zeros((Nf, nw))

    for n in range(nw):
        idx = np.arange(winstart[n], winstart[n] + Nwin - 1)
        data = sig[idx]
        f, s, nu = tsa.multi_taper_psd(data, Fs=Fs, BW=BW, NFFT=nfft)
        S[:, n] = s
    winmid = winstart + round(Nwin / 2.0)
    t = winmid / float(Fs)

    if plot == True:
        plot_specgram(S, t, f, ax)
    elif plot == 'dB':
        plot_specgram(dB(S), t, f, ax)

    return (S, t, f)

def plot_specgram(S, t, f, ax=None):
    if ax == None:
        ax = plt.gca()

    Sm = np.ma.masked_where(np.isnan(S), S)

    mesh = ax.pcolormesh(t, f, Sm, cmap='jet')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('frequency [Hz]')
    #plt.colorbar(mesh, ax=ax)
