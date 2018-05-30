"""Representing responses in a domain agnostic manner."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window, resample
from scipy.io import wavfile


class Response(object):
    """Representation of a linear response in time and frequency domain."""

    def __init__(
        self, fs, fdata=None, tdata=None, isEvenSampled=True, norm=True, unit=None
    ):
        """Create Response from time or frequency data.

        Use `from_time` or `from_freq methods` to create objects of this class!

        Parameters
        ----------
        fs : int
            Sampling frequency in Hertz
        fdata : (ns, nr, nt) complex ndarray, optional
            Single sided amplitude spectra with nt from ns to nr points.
        tdata : (ns, nr, nf) real ndarray, optional
            Time responses with nt from ns to nr points.
        isEvenSampled : bool or None, optional
            If fdata is given, this tells us if the last entry of fdata is the
            Nyquist frequency or not. Must be `None` if tdata is given.
        norm: bool, optional
            If True, sinusoid amplitudes are conserved.

        Raises
        ------
        ValueError
            if neither fdata or tdata are given.

        TODO: remove normalized functionionality because now it is clear what
              the normalization does.

        """
        assert (
            (fdata is not None and tdata is None)
            or (tdata is not None and fdata is None)
        )

        if fdata is not None:
            # fdata is given

            fdata = np.atleast_1d(fdata)
            self._nf = fdata.shape[-1]

            if isEvenSampled:
                self._nt = 2 * (self._nf - 1)
            else:
                self._nt = 2 * self._nf - 1
            self._isEvenSampled = isEvenSampled

            self._set_frequency_data(fdata)
        else:
            # tdata is given

            tdata = np.atleast_1d(tdata)
            self._nt = tdata.shape[-1]
            self._nf = self._nt // 2 + 1
            self._isEvenSampled = (self._nt % 2 == 0)

            self._set_time_data(tdata)

        self._fs = fs
        self._freqs = freq_vector(self._nt, fs)
        self._times = time_vector(self._nt, fs)
        self._time_length = self._nt * 1 / fs
        self._normed = norm
        self._unit = unit

    @classmethod
    def from_time(cls, fs, tdata, **kwargs):
        """Generate Response obj from time response data."""
        tf = cls(fs, tdata=tdata, **kwargs)
        return tf

    @classmethod
    def from_freq(cls, fs, fdata, **kwargs):
        """Generate Response obj from frequency response data."""
        tf = cls(fs, fdata=fdata, **kwargs)
        return tf

    @classmethod
    def from_wav(cls, fps):
        """Import responses from wav files.

        Parameters
        ----------
        fps : list of file paths

        Returns
        -------
        Response
            New Response object with imported time responses.

        """
        fpi = iter(fps)
        fs, data = wavfile.read(next(fpi))
        hlist = [data] + [wavfile.read(fp)[1] for fp in fpi]

        h = np.array(hlist)
        lim_orig = (np.iinfo(data.dtype).min, np.iinfo(data.dtype).max)
        lim_new = (-1., 1.)
        h_float = rescale(h, lim_orig, lim_new).astype(np.double)

        return cls.from_time(fs, h_float)

    @classmethod
    def join(cls, tfs, axis=0, newaxis=True):
        """Concat or stack a set of Responses along a given axis.

        Parameters
        ----------
        tfs : array_like
            List of Responses
        axis : int, optional
            Indice of axis along wich to concatenate / stack TFs.
        newaxis : bool, optional
            If True, do not concatenate but stack arrays along a new axis.

        Returns
        -------
        Response

        Note
        ----
        Transfer functions need to have same sampling rate, length etc.

        """
        joinfunc = np.stack if newaxis else np.concatenate
        tdata = joinfunc([tf.in_time for tf in tfs], axis=axis)
        tf = cls(tfs[0].fs, tdata=tdata, norm=tfs[0]._normed)
        return tf

    @property
    def time_length(self):
        """Length of time response in seconds."""
        return self._time_length

    @property
    def nf(self):  # noqa: D401
        """Number of frequencies in frequency representation."""
        return len(self._freqs)

    @property
    def nt(self):  # noqa: D401
        """Number of taps."""
        return len(self._times)

    @property
    def fs(self):  # noqa: D401
        """Sampling frequency."""
        return self._fs

    @property
    def freqs(self):  # noqa: D401
        """Frequencies."""
        return self._freqs

    @property
    def times(self):  # noqa: D401
        """Times."""
        return self._times

    @property
    def in_time(self):
        """Time domain response.

        Returns
        -------
        (... , n) ndarray
            Real FIR filters.

        """
        if self._in_time is None:
            self._in_time = np.fft.irfft(self._in_freq, n=self._times.size)
        return self._in_time

    @property
    def in_freq(self):
        """Single sided amplitude spectrum.

        Returns
        -------
        (... , n) ndarray
            Complex frequency response.

        """
        if self._in_freq is None:
            self._in_freq = np.fft.rfft(self._in_time)
        return self._in_freq

    def _set_time_data(self, tdata):
        """Set time data without creating new object."""
        assert tdata.shape[-1] == self._nt
        self._in_time = tdata
        self._in_freq = None

    def _set_frequency_data(self, fdata):
        """Set frequency data without creating new object."""
        assert fdata.shape[-1] == self._nf
        self._in_freq = fdata
        self._in_time = None

    def plot(
        self,
        group_delay=False,
        slce=None,
        flim=None,
        dblim=None,
        tlim=None,
        grpdlim=None,
        dbref=1,
        show=False,
        use_fig=None,
        label=None,
        unwrap=False,
        **fig_kw
    ):
        """Plot the response in both domains.

        Parameters
        ----------
        group_delay : bool, optional
            Display group delay instead of phase.
        slce : numpy.lib.index_tricks.IndexExpression
            only plot subset of responses defined by a slice. Last
            dimension (f, t) is always completely taken.
        flim : tuple, optional
            Description
        dblim : None, optional
            Description
        tlim : None, optional
            Description
        dbref : float
            dB reference in magnitude plot
        show : bool, optional
            Description
        fig : matplotlib.pyplot.Figure
            Add data to the axes of a figure.
        label : None, optional
            Description
        unwrap : bool, optional
            Description
        **fig_kw
            Description

        """
        if group_delay:
            unwrap = True

        # append frequency/time dimension to slice
        if slce is None:
            slce = [np.s_[:] for n in range(len(self.in_time.shape))]
        elif isinstance(slce, tuple):
            slce = slce + (np.s_[:],)
        else:
            slce = (slce, np.s_[:])

        unit = " " + self._unit if self._unit else ""

        # move time / frequency axis to first dimension
        freq_plotready = np.rollaxis(self.in_freq[slce], -1).reshape((self.nf, -1))
        time_plotready = np.rollaxis(self.in_time[slce], -1).reshape((self.nt, -1))

        if use_fig is None:
            fig, axes = plt.subplots(nrows=3, constrained_layout=True, **fig_kw)
        else:
            fig = use_fig
            axes = fig.axes

        axes[0].semilogx(
            self.freqs, 20 * np.log10(np.abs(freq_plotready / dbref)), label=label
        )
        axes[0].set_xlabel("Frequency [Hz]")
        axes[0].set_ylabel("Magnitude [dB re {:.2}{}]".format(float(dbref), unit))
        axes[0].set_title("Amplitude response")

        phase = (
            np.unwrap(np.angle(freq_plotready)) if unwrap else np.angle(freq_plotready)
        )

        if group_delay:
            df = self.freqs[1] - self.freqs[0]
            grpd = -np.gradient(phase, df, axis=0)
            axes[1].semilogx(self.freqs, grpd)
            axes[1].set_xlabel("Frequency [Hz]")
            axes[1].set_ylabel("Time [s]")
            axes[1].set_title("Group Delay")
            if grpdlim:
                axes[1].set_ylim(grpdlim)
        else:
            axes[1].semilogx(self.freqs, phase)
            axes[1].set_xlabel("Frequency [Hz]")
            axes[1].set_ylabel("Phase [rad]")
            axes[1].set_title("Phase response")

        axes[2].plot(self.times, time_plotready)
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("")
        axes[2].set_title("Time response")

        for ax in axes:
            ax.grid(True)

        if flim is None:
            flim = (10, self.fs / 2)
        axes[0].set_xlim(flim)
        axes[1].set_xlim(flim)

        if tlim is not None:
            axes[2].set_xlim(tlim)

        if dblim:
            axes[0].set_ylim(dblim)

        if label:
            axes[0].legend()

        if show:
            plt.show()

        return fig

    def time_window(self, startwindow, stopwindow, window="hann", ret_window=False):
        """Apply time windows.

        Parameters
        ----------
        startwindow : None or tuple
            Tuple (t1, t2) with beginning and end times of window opening.
        stopwindow : None or tuple
            Tuple (t1, t2) with beginning and end times of window closing.
        window : string or tuple of string and parameter values, optional
            Desired window to use. See scipy.signal.get_window for a list of
            windows and required parameters.

        Returns
        -------
        in_time: ndarray
            windowed time response
        twindow: ndarray
            applied time window

        """
        n = self.times.size
        twindow = time_window(self.fs, n, startwindow, stopwindow, window=window)
        new_response = self.from_time(self.fs, self.in_time * twindow)

        if ret_window:
            return twindow

        return new_response

    def delay(self, dt, keep_length=True):
        """Delay time response by dt seconds.

        Rounds of to closest integer delay.
        """
        x = delay(self.fs, self.in_time, dt, keep_length=keep_length)
        return self.from_time(self.fs, x)

    def timecrop(self, start, end):
        """Crop time response.

        Parameters
        ----------
        start, end : float
            Start and end times in seconds.

        Returns
        -------
        Response
            New Response object with cropped time.

        Notes
        -----
        Creates new Response object.

        """
        assert start < end

        _, i_start = find_nearest(self.times, start)
        _, i_end = find_nearest(self.times, end)

        h = self.in_time[..., i_start:i_end]

        new_response = self.from_time(self.fs, h)

        return new_response

    def zeropad(self, before, after):
        """Zeropad time response.

        Parameters
        ----------
        before, after : int
            Number of zero samples inserted before and after response.

        Returns
        -------
        Response
            Zeropadded response

        """
        assert before % 1 == 0
        assert after % 1 == 0
        dims = self.in_time.ndim

        pad_width = [(0, 0) for n in range(dims)]
        pad_width[-1] = (int(before), int(after))

        h = np.pad(self.in_time, pad_width, "constant")

        return self.from_time(self.fs, h)

    def zeropad_to_power_of_2(self):
        """Pad time response for length of power of 2.

        Returns
        -------
        Response
            New response object with larger, power of 2 length.

        """
        # https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
        n = 2 ** (self.nt - 1).bit_length()
        return self.zeropad(0, n - self.nt)

    def zeropad_to_length(self, n):
        """Zeropad time response to length.

        Returns
        -------
        Response
            New response object with new length n.

        """
        oldn = self.nt
        assert n >= oldn
        return self.zeropad(0, n - oldn)

    def lowpass_by_frequency_domain_window(self, fstart, fstop):
        """Lowpass response by time domain window."""
        h = lowpass_by_frequency_domain_window(self.fs, self.in_time, fstart, fstop)
        return self.from_time(self.fs, h)

    def resample(self, fs_new, keep_gain=True, window=None):
        """Resample.

        Parameters
        ----------
        fs_new : int
            New sample rate
        keep_gain : bool, optional
            If keep gain is true, normalize such that the gain is the same
            as the original signal.
        window : None, optional
            Passed to scipy.signal.resample.

        Returns
        -------
        Response
            New resampled response object.

        Raises
        ------
        ValueError
            If resulting number of samples would be a non-integer.

        """
        nt_new = fs_new * self.time_length

        if nt_new % 1 != 0:
            raise ValueError(
                "New number of samples must be integer, but is {}".format(nt_new)
            )
        else:
            nt_new = int(nt_new)

        h_new = resample(self.in_time, nt_new, axis=-1, window=window)

        if keep_gain:
            h_new *= self.nt / nt_new

        return self.from_time(fs_new, h_new)

    def export_wav(self, folder, name_fmt="{:02d}.wav", dtype=np.int16):
        """Export response to wave file.

        Parameters
        ----------
        folder : file path
            Save in this folder
        name_fmt : str, optional
            Format string for file names with one placeholder, e.g. 'filt1{:02d}.wav'.

        """
        assert self.in_time.ndim == 2
        assert np.all(np.abs(self.in_time) <= 1.0)

        # convert and scale to new output datatype
        lim_orig = (-1., 1.)
        lim_new = (np.iinfo(dtype).min, np.iinfo(dtype).max)
        data = rescale(self.in_time, lim_orig, lim_new).astype(dtype)

        for i in range(data.shape[0]):
            fp = Path(folder) / name_fmt.format(i + 1)
            wavfile.write(fp, self.fs, data[i])

    @classmethod
    def time_vector(cls, n, fs):
        """Time values of filter with n taps sampled at fs.

        Parameters
        ----------
        n : int
            number of taps in FIR filter
        fs : int
            sampling frequency in Hertz

        Returns
        -------
        (n) ndarray
            times in seconds

        """
        return time_vector(n, fs)

    @classmethod
    def freq_vector(n, fs, sided="single"):
        """Frequency values of filter with n taps sampled at fs up to Nyquist.

        Parameters
        ----------
        n : int
            Number of taps in FIR filter
        fs : int
            Sampling frequency in Hertz

        Returns
        -------
        (n // 2 + 1) ndarray
            Frequencies in Hz

        """
        return freq_vector(n, fs, sided=sided)

####################
# Module functions #
####################


def time_vector(n, fs):
    """Time values of filter with n taps sampled at fs.

    Parameters
    ----------
    n : int
        number of taps in FIR filter
    fs : int
        sampling frequency in Hertz

    Returns
    -------
    (n) ndarray
        times in seconds

    """
    T = 1 / fs
    return np.arange(n, dtype=float) * T  # float against int wrapping


def freq_vector(n, fs, sided="single"):
    """Frequency values of filter with n taps sampled at fs up to Nyquist.

    Parameters
    ----------
    n : int
        Number of taps in FIR filter
    fs : int
        Sampling frequency in Hertz

    Returns
    -------
    (n // 2 + 1) ndarray
        Frequencies in Hz

    """
    # use float against int wrapping
    if sided == "single":
        f = np.arange(n // 2 + 1, dtype=float) * fs / n
    elif sided == "double":
        f = np.arange(n, dtype=float) * fs / n
    else:
        raise ValueError("Invalid value for sided.")

    return f


def time_window(fs, n, startwindow, stopwindow, window="hann"):
    """Create a time window."""
    times = time_vector(n, fs)
    twindow = np.ones(n)

    if startwindow is not None:
        startwindow = list(startwindow)
        if startwindow[0] is None:
            startwindow[0] = 0
        if startwindow[1] is None:
            startwindow[1] = times[-1]
        samples = [find_nearest(times, t)[1] for t in startwindow]
        length = samples[1] - samples[0]
        w = get_window(window, 2 * length, fftbins=False)[:length]
        twindow[:samples[0]] = 0
        twindow[samples[0]:samples[1]] = w

    if stopwindow is not None:
        stopwindow = list(stopwindow)
        if stopwindow[0] is None:
            stopwindow[0] = 0
        if stopwindow[1] is None:
            stopwindow[1] = times[-1]
        samples = [find_nearest(times, t)[1] for t in stopwindow]
        length = samples[1] - samples[0]
        w = get_window(window, 2 * length, fftbins=False)[length:]
        twindow[samples[0] + 1:samples[1] + 1] = w
        twindow[samples[1] + 1:] = 0

    return twindow


def delay(fs, x, dt, keep_length=True, axis=-1):
    """Delay time signal by dt seconds by inserting zeros."""
    dn = int(round(dt * fs))

    zeros_shape = list(x.shape)
    zeros_shape[axis] = dn
    zeros = np.zeros(zeros_shape)

    delayed = np.concatenate((zeros, x), axis=axis)

    if keep_length:
        # slice that takes 0 to ntaps samples along axis
        slc = [slice(None)] * len(x.shape)
        slc[axis] = slice(0, x.shape[axis])
        delayed = delayed[slc]

    return delayed


def lowpass_by_frequency_domain_window(fs, x, fstart, fstop, axis=-1, window='hann'):
    """Lowpass by applying a frequency domain window.

    Parameters
    ----------
    fs : int
        Sampling frequency
    x : array like
        Real time domain signal
    fstart : float
        Starting frequency of window
    fstop : TYPE
        Ending frequency of window
    axis : TYPE, optional
        signal is assumed to be along x[axis]
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter.

    Returns
    -------
    ndarray
        Filtered time signal

    Raises
    ------
    ValueError
        If fstart or fstop don't fit in the frequency range.

    """
    n = x.shape[axis]
    f = freq_vector(n, fs)

    # corresponding indices
    _, start = find_nearest(f, fstart)
    _, stop = find_nearest(f, fstop)

    if not (start and stop):
        raise ValueError("Frequencies are to large.")

    # the window
    window_width = stop - start
    windowed_samples = np.arange(start, stop)

    symmetric_window = get_window(window, 2 * window_width, fftbins=False)
    half_window = symmetric_window[window_width:]

    # frequency domain
    X_windowed = np.fft.rfft(x, axis=axis)
    X_windowed = np.moveaxis(X_windowed, axis, 0)
    X_windowed[windowed_samples] = (
        X_windowed[windowed_samples].T * half_window.T
    ).T  # broadcasting
    X_windowed[stop:] = 0
    X_windowed = np.moveaxis(X_windowed, 0, axis)

    return np.fft.irfft(X_windowed, axis=axis, n=n)


#########
# Utils #
#########


def rescale(x, xlim, ylim):
    """Rescale values to new bounds.

    Parameters
    ----------
    x : ndarray
        Values to rescale
    xmin, xmax : float
        Original value bounds
    ymin, ymax : float
        New value bounds

    Returns
    -------
    ndarray
        Rescaled values

    """
    m = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    c = ylim[1] - m * xlim[1]
    y = m * x + c
    return y


def find_nearest(array, value):
    """Find nearest value in an array and its index.

    Returns
    -------
    value
        Value of nearest entry in array
    idx
        Index of that value

    """
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
