
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



def read_iv_csv(fname, round_30min=False):
    """Read IV-curve file from Sandia.

    Data will also be cleaned to remove 'Unnamed' columns.  The current and voltage values will be
    compressed into their own cell ('current_series' and 'voltage_series').   These values are
    listed in the 'Unnamed' columns that are being removed.

    Parameters
    ----------
    fname: str
        Path to file.

    Returns
    -------
    df: pd.DataFrame
        DataFrame of IV data with 'Unnamed' columns removed.
    """
    df = pd.read_csv(fname)
    ser_list = []
    for index, ser in df.iterrows():
        if np.isfinite(ser['points']):
            ser['current_array'] = ser.values[12: 12 + int(ser['points'])]
            ser['voltage_array'] = ser.values[12 + int(ser['points']): 12 + (2 * int(ser['points']))]
            ser_list.append(ser[['datetime', 'irrad', 'temp', 'string', 'isc', 'voc', 'pmax',
                                 'ipmax', 'vpmax', 'ff', 'points', 'current_array', 'voltage_array']])
#         except ValueError:  # 'points' is NaN, breaks array slicing, skip entry
#             continue
    df = pd.DataFrame(ser_list)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].apply(lambda x: x.date())
    df['time'] = df['datetime'].apply(lambda x: x.time())
    if round_30min:
        df['datetime_rounded'] = df['datetime'].dt.round('30min')
    return df


def plot_iv_info(ser, ax=None, ax2=None, savefig=None):
    if ax is None:
        fig, ax = plt.subplots()

    iv = ax.plot(ser['voltage_array'], ser['current_array'], label='IV')
    mpp = ax.scatter(ser['vpmax'], ser['ipmax'], label='MPP(IV)', zorder=100)

    ff_horizontal = ax.plot([0, ser['vpmax']], [ser['ipmax'], ser['ipmax']], linestyle='--', c='k', alpha=.5)
    ff_vertical = ax.plot([ser['vpmax'], ser['vpmax']], [0, ser['ipmax']], linestyle='--', c='k', alpha=.5)

    ax.set_xlabel('Voltage / V')
    ax.set_ylabel('Current / A')
    ax.set_title(ser['datetime'])

#     i_ratio = ax.plot([0, ser['vpmax']], [ser['isc'], ser['ipmax']], linestyle='--', label='I$_\mathrm{ratio}$')
#     v_ratio = ax.plot([ser['vpmax'], ser['voc']], [ser['ipmax'], 0], linestyle='--', label='V$_\mathrm{ratio}$')

    if ax2 is None:
        ax2 = ax.twinx()

    p_vs_v = ax2.plot(ser['voltage_array'], ser['voltage_array'] * ser['current_array'], c='C1', label='PV')
    mpp_p_vs_v = ax2.scatter(ser['vpmax'], ser['pmax'], c='C1', label='MPP(PV)')

    ax2.set_ylabel('Power / W')

    lines = iv + p_vs_v # + i_ratio + v_ratio
    labels = [l.get_label() for l in lines]
    legend = ax.legend(lines, labels, loc=('lower center'))

    if savefig is not None:
        plt.savefig(savefig, dpi=300)
    return ax


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

