

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn import linear_model

import utils

def main():
    pass


class IVCurve(object):

    """
    This class is used for processing, analyzing, and extracting information for current-voltage (IV) curves.
    Current work is focused on building out the parameter extraction for modeling methods (LFM, SAPM).
    """

    def __init__(self, voltage, current, temp=None, irrad=None, name=None, low_voltage_cutoff=25, min_pts=25):
        """Initialize object.

        Parameters
        ----------
        voltage: array-like
        current: array-like
        name: object
            Name for sample.
        low_voltage_cutoff: float
            Voltages near Isc can be very noisy (based on data used).  Ignore any voltages < this value.
        min_pts: int
            Require a minimum number of points to exist for analysis.  This is after removing low voltage points.
        """
        # voltage = np.asarray(voltage).flatten().astype(float)
        # current = np.asarray(current).flatten().astype(float)
        mask = voltage > low_voltage_cutoff
        x = voltage[mask].astype(float)
        y = current[mask].astype(float)

        # assure adequate amount of data points
        if len(x) < min_pts or len(y) < min_pts:
            raise ValueError('Not enough points in IV curve.')

        # sorting IV curve by voltage (needed for interpolation)
        xy = np.column_stack((x, y))
        xy = xy[xy[:, 0].argsort()]
        x = xy[:, 0]
        y = xy[:, 1]

        self.v_ = x
        self.i_ = y

        if name is not None:
            self.name_ = name
        else:
            self.name_ = 0

        self.temp_ = temp
        self.irrad_ = irrad

        self.interp_func_ = None

    def is_smooth(self, pct_change=.1, start=10, end=10):
        """Check how smooth the interpolated curve is.  A curve is adequately smooth if the maximum percent change
        between adjacent points is < pct_change.  This could/should be made more robust.

        Parameters
        ----------
        pct_change: float
            Maximum allowed percent change to determine smoothness.
        start: int
            Check smoothness of curve from array[start:] (ignore first 'n' points).
        end:int
            Check smoothness of curve up to array[:-end] (ignore last 'n' points).

        Returns
        -------
        bool
        """
        v, i = self.smooth()
        v = v[start:-end]
        i = i[start:-end]
        return np.max(np.abs(np.diff(i) / i[:-1])) < pct_change

    def extract_components(self, isc_lim=.1, voc_lim=.1, mode='all'):# , with_models=False):
        """Extract Common IV curve characteristics.  Optionally returns parameters needed for modeling with the
        Loss Factor Model and the Sanida Array Performance Model.

        Isc and Voc are found by performing a linear regression on endpoints of the data.
        Rs and Rsh are determined used slopes from the above regression.

        The characterisitics are:
            Isc: short-circuit current
            Voc: open-circuit voltage
            Ipmax: current at maximum power
            Vpmax: voltage at maximum power
            Rs: series resistance
            Rsh: shunt resistance

        Additional values for LFM:
            Vr: voltage at intersection of Rsh and Rs tangent line
            Ir: current at intersection of Rsh and Rs tangent line

        Additional values for SAPM:
            Ix: current at 1/2 Voc
            Ixx: current at 1/2 (Voc + Vpmax)

        Parameters
        ----------
        isc_lim: float or int
            Percent or number of data points to calculate Isc and Rsh with.
        voc_lim: float or int
            Percent or number of data points to calculate Voc and Rs with.
        mode: str
            Return extra parameters for specific performance models (lfm, sapm).

        Returns
        -------
        params: dict
            Key and value pairs are described above.
        """
        v, i = self.smooth()

        # maximum power point
        pmax = np.max((v * i))
        mpp_idx = np.argmax((v * i))
        vpmax = v[mpp_idx]
        ipmax = i[mpp_idx]

        # isc and rsh
        if type(isc_lim) == float:
            isc_size = int(len(i) * isc_lim)
        else:
            isc_size = isc_lim
        isc_lm = linear_model.LinearRegression().fit(v[:isc_size].reshape(-1, 1), i[:isc_size].reshape(-1, 1))
        isc = isc_lm.predict(np.asarray([0]).reshape(-1, 1))[0][0]
        rsh = isc_lm.coef_[0][0] * -1

        # voc and rs
        if type(voc_lim) == float:
            voc_size = int(len(v) * voc_lim)
        else:
            voc_size = voc_lim
        voc_lm = linear_model.LinearRegression().fit(i[::-1][:voc_size].reshape(-1, 1),
                                                     v[::-1][:voc_size].reshape(-1, 1))
        voc = voc_lm.predict(np.asarray([0]).reshape(-1, 1))[0][0]
        rs = voc_lm.coef_[0][0] * -1

        # fill factor
        ff = (ipmax * vpmax) / (isc * voc) * 100

        params = {'pmax': pmax,
                  'vpmax': vpmax,
                  'ipmax': ipmax,
                  'voc': voc,
                  'isc': isc,
                  'rs': rs,
                  'rsh': rsh,
                  'ff': ff}

        # LFM
        if mode.lower() in ('all', 'lfm'):
            isc_lm_m, isc_lm_b = isc_lm.coef_[0], isc_lm.intercept_[0]
            voc_lm_m, voc_lm_b = voc_lm.coef_[0], voc_lm.intercept_[0]

            vr = ((isc_lm_b * voc_lm_m) + voc_lm_b) / (1 - (isc_lm_m * voc_lm_m))
            vr = vr[0]
            ir = isc_lm.predict(np.asarray([vr]).reshape(-1, 1))[0][0]

            params['vr'] = vr
            params['ir'] = ir

        # SAPM
        if mode.lower() in ('all', 'sapm'):
            ix = self.smooth(np.asarray([.5 * voc]))[1]
            ixx = self.smooth(np.asarray([.5 * (voc + vpmax)]))[1]
            params['ix'] = ix
            params['ixx'] = ixx

        # if with_models:
        #     params['isc_lm'] = isc_lm
        #     params['voc_lm'] = voc_lm

        return params

    def calc_lfm_params(self, rvoc, risc, rvpmax, ripmax, alpha, beta):
        """Caluclate parameters for Loss Factor Model (LFM).  Measured parameters will be extracted from IV curve while
        reference values must be supplied.

        Parameters
        ----------
        rvoc: float
            Reference open-circuit voltage (@STC).
        risc: float
            Reference short-circuit current (@STC).
        rvpmax: float
            Reference maximum power point voltage (@STC).
        ripmax: float
            Reference maximum power point current (@STC).
        alpha: float 
            Temperature coefficient for current.
        beta: float
            Temperature coefficient for voltage.

        Returns
        -------
        lfm_params: dict
            Parameters for modeling PV system using the LFM.
        """
        if self.irrad_ is None or self.temp_ is None:
            raise ValueError('Must set member variables self.temp_ and self.irrad_ in order to calculate LFM parameters.')
        components = self.extract_components(mode='all')
        lfm_params = {}
        lfm_params['nisct'] = components['isc'] / risc / self.irrad_ * (1 + alpha * (25 - self.temp_))
        lfm_params['nrsc'] = components['ir'] / components['isc']
        lfm_params['nimp'] = components['ipmax'] * risc / components['ir'] / ripmax
        lfm_params['nroc'] = components['vr'] / components['voc']
        lfm_params['nvmp'] = components['vpmax'] * rvoc / components['vr'] / rvpmax
        lfm_params['nvoct'] = components['voc'] / rvoc * (1 + beta * (25 - self.temp_))
        lfm_params['pimp'] = lfm_params['nisct'] * lfm_params['nrsc'] * lfm_params['nimp'] * ripmax * \
                             (self.irrad_ / (1 + alpha * (25 - self.temp_)))
        lfm_params['pvmp'] = lfm_params['nvmp'] * lfm_params['nroc'] * lfm_params['nvoct'] * rvpmax / \
                             (1 + beta * (25 - self.temp_))
        return lfm_params

    def normalize_curve(self, set_member=False, by='isc_voc', spline_kwargs={}):
        """Normalize IV curve by Isc/Voc or by MPP.  Data will be re-interpolated if set_member is True.

        Parameters
        ----------
        set_member: bool
            Reset member variables self.v_, self.i_ with normalized values.
        by: str
            Must be either 'isc_voc' or 'mpp'.
        spline_kwargs: dict
            Parameters for smoothing spline.

        Returns
        -------
        v: np.asarray
            Voltages
        i: np.asarray
            Currents
        """
        assert False, 'Method needs work/reimplementation.'
        components = self.extract_components()
        if by == 'isc_voc':
            i_scaler = components['isc']
            v_scaler = components['voc']
        elif by == 'mpp':
            i_scaler = components['ipmax']
            v_scaler = components['vpmax']
        else:
            raise ValueError("Argument 'by' must be either 'isc_voc' or 'mpp'.")

        v, i = self.smooth()

        v = v / v_scaler
        i = i / i_scaler

        if set_member:
            self.v_ = v
            self.i_ = i
            if len(spline_kwargs) > 0:
                self.interpolate(**spline_kwargs)
            else:
                self.interpolate()

        return v, i

    def interpolate(self, pct_change_allowed=0.1, spline_kwargs={'s': 0.025}):
        """Get interpolation function.

        Parameters
        ----------
        spline_kwargs: dict
            Keywords for spline (see scipy.interpolate.splrep).

        Returns
        -------
        self
        """
        pct_change = np.abs(np.diff(self.i_)) / self.i_[:-1]
        pct_change = np.insert(pct_change, 0, 0)
        v = self.v_[pct_change < pct_change_allowed]
        i = self.i_[pct_change < pct_change_allowed]
        self.interp_func_ = interpolate.splrep(v, i, **spline_kwargs)
        return self

    def smooth(self, v=None, raw_v=False, der=0, npts=250):
        """Return smoothed IV curve.  If self.interpolate() is not called first, default interpolation will be
        used (see interpolate method).

        Optionally pass specific voltage(s) (v=your_array), use the measured voltage (raw_v=True), or generate
        npts voltage values between endpoints of measured voltages.

        Parameters
        ----------
        v: np.array
            Specific voltage value(s) to calculate current for.  If None, will be skipped.
        raw_v: bool
            If True, use measured voltage values (self.v_).  If False, generate points between first and last elements
            of self.v_.
        der: int
            Order of derivative.
        npts: int
            Number of points to use in IV curve.  Ignored if raw_v is True.

        Returns
        -------
        v_smooth, i_smooth: np.array, np.array
            Evenly spaced voltages and smoothed current values at each point.
        """
        if self.interp_func_ is None:
            self.interpolate()

        if raw_v:
            v_smooth = self.v_
        elif v is not None:
            v_smooth = v
        else:
            v_smooth = np.linspace(self.v_[0], self.v_[-1], npts)
        i_smooth = interpolate.splev(v_smooth, self.interp_func_, der=der)

        return v_smooth, i_smooth

    def locate_mismatch(self, min_peak=-.02, dv_neighbor_cutoff=0,
                        low_v_cutoff=0, vis=False, vis_with_deriv=True, verbose=False,
                        smooth_window=5, smooth_cutoff=1e-4):
        """Find and return points of mismatch in IV curve.

        Mismatches will be identified by finding peaks near zero in the dI/dV curve.

        Parameters
        ----------
        min_peak: float
            Minimum value for a peak.
        dv_neighbor_cutoff: int
            This cutoff is a smoothing parameter for points that may be in the same mismatch region.
            If the index of a point is <= to it's neighbors, they are considered to be the same mismatch region.
        low_v_cutoff: float
            Remove mismatching very close to 0V.
        vis: bool
            Generate visualization.
        vis_with_deriv: bool
            Include derivative in visualization.
        verbose: bool
            Return extra information.
        normalize: bool
            Normalize mismatch_pts by Isc and Voc.  If verbose is True, the smoothed IV curves and derivatives will not
            be normalized.  Normalization will also not be reflected in plots (if vis is True).
        smooth_window: int
            Window size to check smoothness of peaks.  Window defined as [i-smooth_window:i+smooth_window].
        smooth_cutoff: slope
            If smoothness of derivative near peaks is < smooth_cutoff, the peak is ingored.

        Returns
        -------
        mismatch_pts: np.array
            [[voltage, current],
             [...    , ...    ]]  where mismatching is detected.
        components: dict with following key/values:
            v, optional: np.array
                Evenly spaced voltages used to smooth curve.  Returned if verbose == True.
            i, optional: np.array
                Smoothed current values from interpolation.  Returned if verbose == True.
            di_dv, optional: np.array
                Smoothed first derivative from interpolation.  Returned if verbose == True.
            d2i_dv2, optional: np.array
                Smoothed second derivative from interpolation.  Returned if verbose == True.
        """
        # get smoothed curve to work with
        v, i = self.smooth()
        _, di_dv = self.smooth(der=1)
        _, d2i_dv2 = self.smooth(der=2)

        # mismatches = np.isclose(di_dv, 0, atol=atol) & np.less(d2i_dv2, 0)
        mismatches = utils.get_local_extrema(di_dv, cutoff=min_peak)

        # isolate regions of mismatch - consider peaks close together as same peak
        indices = mismatches
        vdiffs = np.insert(np.diff(v[indices]), 0, 0)
        subgroups = []
        group = []
        for ind, vdiff in zip(indices, vdiffs):
            if vdiff <= dv_neighbor_cutoff:
                group.append(ind)
            else:
                subgroups.append(group[:])
                group = [ind]
        subgroups.append(group)

        # construct list of mismatch points
        mismatch_pts = []
        for sg in subgroups:
            if not sg:
                continue
            max_deriv_idx = np.argmax(di_dv[sg]) + sg[0]
            window = di_dv[max_deriv_idx - smooth_window: max_deriv_idx + smooth_window]
            if np.sqrt(np.mean(np.diff(window)**2)) < smooth_cutoff:
                continue
            if v[max_deriv_idx] > low_v_cutoff:
                mismatch_pts.append((v[max_deriv_idx], i[max_deriv_idx]))
        mismatch_pts = np.asarray(mismatch_pts)

        if vis:
            self.mismatch_vis_(v, i, di_dv, d2i_dv2, mismatch_pts, indices, vis_with_deriv=vis_with_deriv)

        if verbose:
            return mismatch_pts, {'v': v, 'i': i, 'di_dv': di_dv, 'd2i_dv2': d2i_dv2}
        else:
            return mismatch_pts

    def normalized_mismatch(self):
        """Normalize mismatch points by Isc/Voc.

        Returns
        -------
        normed_mismatch: list of tuple
            Normalized value pairs.
        """
        raise NotImplementedError('Normalization will be done using LFM parameters.')
        mismatch = self.locate_mismatch()
        components = self.extract_components()
        v = np.asarray([a[0] for a in mismatch])
        i = np.asarray([a[1] for a in mismatch])
        v = v / components['voc']
        i = i / components['isc']
        normed_mismatch = [(vv, ii) for vv, ii in zip(v, i)]
        return normed_mismatch

    def mismatch_vis_(self, v, i, di_dv, d2i_dv2, mismatch_pts, indices, vis_with_deriv=True):
        """Visualize mismatch points and local maxima of dI/dV.  Most conveniently used
        via IVCurve.locate_mismatch(..., vis=True, ...).

        Parameters
        ----------
        v: np.array
            Voltages.
        i: np.array
            Currents.
        di_dv: np.array
            First derivative of IV curve.
        d2i_dv2: np.array
            Second derivative of IV curve.
        mismatch_pts: np.array
            Approximate mismatch locations.
        indices: np.array
            Local maxima voltage indices.

        Returns
        -------
        None
        """
        if vis_with_deriv:
            fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        else:
            fig, axes = plt.subplots()

        if vis_with_deriv:
            ax = axes[0]
        else:
            ax = axes

        ax.plot(self.v_, self.i_, label='raw data')
        ax.plot(v, i, label='smoothed data')
        if len(mismatch_pts) > 0:
            ax.scatter(mismatch_pts[:, 0], mismatch_pts[:, 1], label='approx mismatch',
                       marker='o', facecolor='none', edgecolor='k', linewidth=2, s=400, alpha=1, zorder=-1)
        # ax.scatter(mismatch_x_pts, mismatch_y_pts, label='approx mismatch', marker='+', c='k',
        #            linewidth=2, s=300,alpha=1, zorder=-1)
        _ = ax.legend(loc='lower left')
        _ = ax.set_title('IV profile (sample {})'.format(self.name_))
        # _ = ax.set_xlabel('Voltage / V')
        _ = ax.set_ylabel('Current / A')

        if vis_with_deriv:
            ax = axes[1]
            ax.plot(v, di_dv, label=r'$\frac{ d\mathrm{I} }{ d\mathrm{V} }$')
            # ax.plot(v, d2i_dv2, label=r'$\frac{ d^2\mathrm{I} }{ d\mathrm{V^2} }$')
            ax.scatter(v[indices], di_dv[indices], label='local maxima', alpha=1, marker='o', facecolor='none',
                       edgecolor='k', linewidth=2, s=400, zorder=-1)
            _ = ax.legend(loc='lower left')
            _ = ax.set_title('Derivative IV profile (sample {})'.format(self.name_))
            _ = ax.set_xlabel('Voltage / V')
            _ = ax.set_ylabel('Current / A')

        _ = fig.tight_layout()

    def plot(self, ax=None):
        """Generate IV curve plot.  Will show raw data points and smoothed function.  MPP will be marked.

        Returns
        -------
        None
        """
        v, i = self.smooth()
        components = self.extract_components(mode='all')

        if ax is None:
            fig, ax = plt.subplots()
            fig.tight_layout()

        ax.plot(self.v_, self.i_, label='raw data')
        ax.plot(v, i, label='smoothed data')
        ax.scatter(components['vpmax'], components['ipmax'], label='MPP', c='k', zorder=100)
        ax.plot([0, components['vpmax']], [components['ipmax'], components['ipmax']], linestyle='--', c='k', alpha=.5)
        ax.plot([components['vpmax'], components['vpmax']], [0, components['ipmax']], linestyle='--', c='k', alpha=.5)

        ax.legend()
        ax.set_title('IV profile (sample {})'.format(self.name_))
        ax.set_xlabel('Voltage / V')
        ax.set_ylabel('Current / A')


if __name__ == '__main__':
    main()
