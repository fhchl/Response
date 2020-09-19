"""Your handy frequency and impulse response processing object.

[![](https://img.shields.io/pypi/l/response.svg?style=flat)](https://pypi.org/project/response/)
[![](https://img.shields.io/pypi/v/response.svg?style=flat)](https://pypi.org/project/response/)
[![travis-ci](https://travis-ci.org/fhchl/Response.svg?branch=master)](https://travis-ci.org/fhchl/Response)
[![codecov](https://codecov.io/gh/fhchl/Response/branch/master/graph/badge.svg)](https://codecov.io/gh/fhchl/Response)

This module supplies the `Response` class: an abstraction of frequency and
impulse responses and a set of handy methods for their processing. It implements a
[fluent interface][1] for chaining the processing commands.

Find the documentation [here][2] and the source code on [GitHub][3].

```python
import numpy as np
from response import Response

fs = 48000  # sampling rate
T = 0.5     # length of signal
# a sine at 100 Hz
t = np.arange(int(T * fs)) / fs
x = np.sin(2 * np.pi * 100 * t)
# Do chain of processing
r = (
    Response.from_time(fs, x)
    # time window at the end and beginning
    .time_window((0, 0.1), (-0.1, None), window="hann")  # equivalent to Tukey window
    # zeropad to one second length
    .zeropad_to_length(fs * 1)
    # circular shift to center
    .circdelay(T / 2)
    # resample with polyphase filter, keep gain of filter
    .resample_poly(500, window=("kaiser", 0.5), normalize="same_amplitude")
    # cut 0.2s at beginning and end
    .timecrop(0.2, -0.2)
    # apply frequency domain window
    .freq_window((0, 90), (110, 500))
)
# plot magnitude, phase and time response
r.plot(show=True)
# real impulse response
r.in_time
# complex frequency response
r.in_freq
# and much more ...
```

[1]: https://en.wikipedia.org/wiki/Fluent_interface
[2]: https://fhchl.github.io/Response/
[3]: https://github.com/fhchl/Response

"""

import warnings
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, lfilter, resample, resample_poly, tukey, welch


class Response(object):
    """Representation of a linear response in time and frequency domain."""

    def __init__(self, fs, fdata=None, tdata=None, isEvenSampled=True):
        """Create Response from time or frequency data.

        Use `from_time` or `from_freq methods` to create objects of this class!

        Parameters
        ----------
        fs : int
            Sampling frequency in Hertz
        fdata : (..., nt) complex ndarray, optional
            Single sided frequency spectra with nt from ns to nr points.
        tdata : (..., nf) real ndarray, optional
            Time responses with nt from ns to nr points.
        isEvenSampled : bool or None, optional
            If fdata is given, this tells us if the last entry of fdata is the
            Nyquist frequency or not. Must be `None` if tdata is given.

        Raises
        ------
        ValueError
            if neither fdata or tdata are given.

        """
        assert float(fs).is_integer()

        if fdata is not None and tdata is None:
            fdata = np.atleast_1d(fdata)
            self._nf = fdata.shape[-1]

            if isEvenSampled:
                self._nt = 2 * (self._nf - 1)
            else:
                self._nt = 2 * self._nf - 1
            self._isEvenSampled = isEvenSampled

            self.__set_frequency_data(fdata)
        elif tdata is not None and fdata is None:
            assert np.all(np.imag(tdata) == 0), "Time data must be real."
            tdata = np.atleast_1d(tdata)
            self._nt = tdata.shape[-1]
            self._nf = self._nt // 2 + 1
            self._isEvenSampled = self._nt % 2 == 0

            self.__set_time_data(tdata)
        else:
            raise ValueError("One and only one of fdata and tdata must be given.")

        self._fs = int(fs)
        self._freqs = freq_vector(self._nt, fs)
        self._times = time_vector(self._nt, fs)
        self._time_length = self._nt * 1 / fs
        self.df = self._freqs[1]  # frequency resolution
        self.dt = self._times[1]  # time resolution

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
        fps : list
            File paths of all wav files.

        Returns
        -------
        Response
            New Response object with imported time responses.

        """
        fpi = iter(fps)
        fs, data = wavfile.read(next(fpi))
        hlist = [data] + [wavfile.read(fp)[1] for fp in fpi]

        h = np.array(hlist)
        if data.dtype in [np.uint8, np.int16, np.int32]:
            lim_orig = (np.iinfo(data.dtype).min, np.iinfo(data.dtype).max)
            lim_new = (-1.0, 1.0)
            h = _rescale(h, lim_orig, lim_new).astype(np.double)

        return cls.from_time(fs, h)

    @classmethod
    def new_dirac(cls, fs, T=None, n=None, nch=(1,)):
        """Generate new allpass / dirac response."""
        nch = np.atleast_1d(nch)
        if T is not None:
            nt = round(fs * T)
        else:
            nt = n
        h = np.zeros((*nch, nt))
        h[..., 0] = 1
        return cls.from_time(fs, h)

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
        return cls.from_time(tfs[0].fs, tdata)

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
        """Single sided frequency spectrum.

        Returns
        -------
        (... , n) ndarray
            Complex frequency response.

        """
        if self._in_freq is None:
            self._in_freq = np.fft.rfft(self._in_time)
        return self._in_freq

    @property
    def amplitude_spectrum(self):
        """Amplitude spectrum."""
        X = self.in_freq / self.nt

        if self.nt % 2 == 0:
            # zero and nyquist element only appear once in complex spectrum
            X[..., 1:-1] *= 2
        else:
            # there is no nyquist element
            X[..., 1:] *= 2

        return X

    def __set_time_data(self, tdata):
        """Set time data without creating new object."""
        assert tdata.shape[-1] == self._nt
        self._in_time = tdata
        self._in_freq = None

    def __set_frequency_data(self, fdata):
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
        unwrap_phase=False,
        logf=True,
        third_oct_f=True,
        plot_kw={},
        **fig_kw,
    ):
        """Plot the response in both domains.

        Parameters
        ----------
        group_delay : bool, optional
            Display group delay instead of phase.
        slce : numpy.lib.index_tricks.IndexExpression
            only plot subset of responses defined by a slice. Last
            dimension (frequency or time) is always completely taken.
        flim : tuple or None, optional
            Frequency axis limits as tuple `(lower, upper)`
        dblim : tuple or None, optional
            Magnitude axis limits as tuple `(lower, upper)`
        tlim : tuple or None, optional
            Time axis limits as tuple `(lower, upper)`
        grpdlim: tuple or None, optional
            Group delay axis limit as tuple `(lower, upper)`
        dbref : float
            dB reference in magnitude plot
        show : bool, optional
            Run `matplotlib.pyplot.show()`
        use_fig : matplotlib.pyplot.Figure
            Reuse an existing figure.
        label : None, optional
            Description
        unwrap_phase : bool, optional
            unwrap phase in phase plot
        logf : bool, optional
            If `True`, use logarithmic frequency axis.
        third_oct_f: bool, optional
            Label frequency axis with third octave bands.
        plot_kw : dictionary, optional
            Keyword arguments passed to the `plt.plot` commands.
        **fig_kw
            Additional options passe to figure creation.

        """
        if use_fig is None:
            fig_kw = {**{"figsize": (10, 10)}, **fig_kw}
            fig, axes = plt.subplots(nrows=3, constrained_layout=True, **fig_kw)
        else:
            fig = use_fig
            axes = fig.axes

        self.plot_magnitude(
            use_ax=axes[0],
            slce=slce,
            dblim=dblim,
            flim=flim,
            dbref=dbref,
            label=label,
            plot_kw=plot_kw,
            logf=logf,
            third_oct_f=third_oct_f,
        )
        if group_delay:
            self.plot_group_delay(
                use_ax=axes[1],
                slce=slce,
                flim=flim,
                ylim=grpdlim,
                plot_kw=plot_kw,
                logf=logf,
                third_oct_f=third_oct_f,
            )
        else:
            self.plot_phase(
                use_ax=axes[1],
                slce=slce,
                flim=flim,
                plot_kw=plot_kw,
                unwrap=unwrap_phase,
                logf=logf,
                third_oct_f=third_oct_f,
            )
        self.plot_time(
            use_ax=axes[2], tlim=tlim, slce=slce, plot_kw=plot_kw
        )

        if show:
            plt.show()

        return fig

    def plot_magnitude(
        self,
        use_ax=None,
        slce=None,
        dblim=None,
        flim=None,
        dbref=1,
        label=None,
        plot_kw={},
        logf=True,
        third_oct_f=True,
        **fig_kw,
    ):
        """Plot magnitude response."""
        # TODO: compute db limits similar to librosa.amplitude_to_db / power_to_db
        if use_ax is None:
            fig_kw = {**{"figsize": (10, 5)}, **fig_kw}
            fig, ax = plt.subplots(nrows=1, constrained_layout=True, **fig_kw)
        else:
            ax = use_ax
            fig = ax.get_figure()

        # append frequency/time dimension to slice
        if slce is None:
            slce = [np.s_[:] for n in range(len(self.in_time.shape))]
        elif isinstance(slce, tuple):
            slce = slce + (np.s_[:],)
        else:
            slce = (slce, np.s_[:])

        # move time / frequency axis to first dimension
        freq_plotready = np.rollaxis(self.in_freq[tuple(slce)], -1).reshape(
            (self.nf, -1)
        )

        plotf = ax.semilogx if logf else ax.plot
        plotf(
            self.freqs,
            20 * np.log10(np.abs(freq_plotready / dbref)),
            label=label,
            **plot_kw,
        )
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_title("Frequency response")
        ax.grid(True)

        if flim is None:
            lowlim = min(10, self.fs / 2 / 100)
            flim = (lowlim, self.fs / 2)
        ax.set_xlim(flim)

        if dblim is not None:
            ax.set_ylim(dblim)

        if label is not None:
            ax.legend()

        if third_oct_f:
            _add_octave_band_xticks(ax)

        return fig

    def plot_phase(
        self,
        use_ax=None,
        slce=None,
        flim=None,
        label=None,
        unwrap=False,
        ylim=None,
        plot_kw={},
        logf=True,
        third_oct_f=True,
        **fig_kw,
    ):
        """Plot phase response."""
        if use_ax is None:
            fig_kw = {**{"figsize": (10, 5)}, **fig_kw}
            fig, ax = plt.subplots(nrows=1, constrained_layout=True, **fig_kw)
        else:
            ax = use_ax
            fig = ax.get_figure()

        # append frequency/time dimension to slice
        if slce is None:
            slce = [np.s_[:] for n in range(len(self.in_time.shape))]
        elif isinstance(slce, tuple):
            slce = slce + (np.s_[:],)
        else:
            slce = (slce, np.s_[:])

        # move time / frequency axis to first dimension
        freq_plotready = np.rollaxis(self.in_freq[tuple(slce)], -1).reshape(
            (self.nf, -1)
        )
        phase = (
            np.unwrap(np.angle(freq_plotready)) if unwrap else np.angle(freq_plotready)
        )

        plotf = ax.semilogx if logf else ax.plot
        plotf(self.freqs, phase, label=label, **plot_kw)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Phase [rad]")
        ax.set_title("Phase response")
        ax.grid(True)

        if flim is None:
            lowlim = min(10, self.fs / 2 / 100)
            flim = (lowlim, self.fs / 2)
        ax.set_xlim(flim)
        if ylim:
            ax.set_ylim(ylim)

        if label is not None:
            ax.legend()

        if third_oct_f:
            _add_octave_band_xticks(ax)

        return fig

    def plot_time(
        self,
        use_ax=None,
        slce=None,
        tlim=None,
        ylim=None,
        label=None,
        plot_kw={},
        **fig_kw,
    ):
        """Plot time response."""
        if use_ax is None:
            fig_kw = {**{"figsize": (10, 5)}, **fig_kw}
            fig, ax = plt.subplots(nrows=1, constrained_layout=True, **fig_kw)
        else:
            ax = use_ax
            fig = ax.get_figure()

        # append frequency/time dimension to slice
        if slce is None:
            slce = [np.s_[:] for n in range(len(self.in_time.shape))]
        elif isinstance(slce, tuple):
            slce = slce + (np.s_[:],)
        else:
            slce = (slce, np.s_[:])

        time_plotready = np.rollaxis(self.in_time[tuple(slce)], -1).reshape(
            (self.nt, -1)
        )

        ax.plot(self.times, time_plotready, label=label, **plot_kw)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("")
        ax.set_title("Time response")
        ax.grid(True)

        if tlim:
            ax.set_xlim(tlim)
        if ylim:
            ax.set_ylim(ylim)

        if label is not None:
            ax.legend()

        return fig

    def plot_group_delay(
        self,
        use_ax=None,
        slce=None,
        flim=None,
        label=None,
        ylim=None,
        plot_kw={},
        logf=True,
        third_oct_f=True,
        **fig_kw,
    ):
        """Plot group delay."""
        if use_ax is None:
            fig_kw = {**{"figsize": (10, 5)}, **fig_kw}
            fig, ax = plt.subplots(nrows=1, constrained_layout=True, **fig_kw)
        else:
            ax = use_ax
            fig = ax.get_figure()

        # append frequency/time dimension to slice
        if slce is None:
            slce = [np.s_[:] for n in range(len(self.in_time.shape))]
        elif isinstance(slce, tuple):
            slce = slce + (np.s_[:],)
        else:
            slce = (slce, np.s_[:])

        # move time / frequency axis to first dimension
        freq_plotready = np.rollaxis(self.in_freq[tuple(slce)], -1).reshape(
            (self.nf, -1)
        )

        df = self.freqs[1] - self.freqs[0]
        # TODO: use scipy.signal.group_delay here as below has problem at larger delays
        grpd = -np.gradient(np.unwrap(np.angle(freq_plotready)), df, axis=0)

        plotf = ax.semilogx if logf else ax.plot
        plotf(self.freqs, grpd, label=label, **plot_kw)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Delay [s]")
        ax.set_title("Group Delay")
        ax.grid(True)

        if flim is None:
            lowlim = min(10, self.fs / 2 / 100)
            flim = (lowlim, self.fs / 2)
        ax.set_xlim(flim)

        if ylim:
            ax.set_ylim(ylim)

        if label is not None:
            ax.legend()

        if third_oct_f:
            _add_octave_band_xticks(ax)

        return fig

    def plot_power_in_bands(
        self, bands=None, use_ax=None, barkwargs={}, avgaxis=None, dbref=1, **figkwargs
    ):
        """Plot signal's power in bands.

        Parameters
        ----------
        bands : list or None, optional
            List of tuples (f_center, f_lower, f_upper). If `None`, use third octave
            bands.
        use_ax : matplotlib.axis.Axis or None, optional
            Plot into this axis.
        barkwargs : dict
            Keyword arguments to `axis.bar`
        avgaxis : int, tuple or None
            Average power over these axes.
        dbref : float
            dB reference.
        **figkwargs
            Keyword arguments passed to plt.subplots

        Returns
        -------
        P : ndarray
            Power in bands
        fc : ndarray
            Band frequencies
        fig : matplotlib.figure.Figure
            Figure

        """
        P, fc = self.power_in_bands(bands=bands, avgaxis=avgaxis)

        nbands = P.shape[-1]
        P = np.atleast_2d(P).reshape((-1, nbands))

        if use_ax is None:
            fig, ax = plt.subplots(**figkwargs)
        else:
            ax = use_ax
            fig = ax.get_figure()

        xticks = range(1, nbands + 1)
        for i in range(P.shape[0]):
            ax.bar(xticks, 10 * np.log10(P[i] / dbref ** 2), **barkwargs)
        ax.set_xticks(xticks)
        ax.set_xticklabels(["{:.0f}".format(f) for f in fc], rotation="vertical")
        ax.grid(True)
        ax.set_xlabel("Band's center frequencies [Hz]")
        ax.set_ylabel("Power [dB]")

        return (P, fc, fig)

    def time_window(self, startwindow, stopwindow, window="hann"):
        """Apply time domain windows.

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
        Response
            Time windowed response object

        """
        n = self.times.size
        twindow = _time_window(self.fs, n, startwindow, stopwindow, window=window)
        new_response = self.from_time(self.fs, self.in_time * twindow)

        return new_response

    def freq_window(self, startwindow, stopwindow, window="hann"):
        """Apply frequency domain window.

        Parameters
        ----------
        startwindow : None or tuple
            Tuple (t1, t2) with beginning and end frequencies of window opening.
        stopwindow : None or tuple
            Tuple (t1, t2) with beginning and end frequencies of window closing.
        window : string or tuple of string and parameter values, optional
            Desired window to use. See scipy.signal.get_window for a list of
            windows and required parameters.

        Returns
        -------
        Response
            Frequency windowed response object

        """
        n = self.times.size
        fwindow = _freq_window(self.fs, n, startwindow, stopwindow, window=window)
        new_response = self.from_freq(self.fs, self.in_freq * fwindow)

        return new_response

    def window_around_peak(self, tleft, tright, alpha, return_window=False):
        """Time window each impulse response around its peak value.

        Parameters
        ----------
        tleft, tright : float
            Window starts `tleft` seconds before and ends `tright` seconds after maximum
            of impulse response.
        alpha : float
            `alpha` parameter of `scipy.signal.tukey` window.
        return_window : bool, optional
            Also return used time window

        Returns
        -------
        Response
            Time windowed response object.
        ndarray
            Time window, if `return_window` is `True`.

        """
        window = _construct_window_around_peak(
            self.fs, self.in_time, tleft, tright, alpha=alpha
        )

        if return_window:
            return self.from_time(self.fs, self.in_time * window), window

        return self.from_time(self.fs, self.in_time * window)

    def delay(self, dt, keep_length=True):
        """Delay time response by dt seconds.

        Rounds of to closest integer delay.
        """
        x = delay(self.fs, self.in_time, dt, keep_length=keep_length)
        return self.from_time(self.fs, x)

    def circdelay(self, dt):
        """Delay by circular shift.

        Rounds of to closest integer delay.
        """
        x = self.in_time
        n = int(round(dt * self.fs))
        shifted = np.roll(x, n, axis=-1)

        return self.from_time(self.fs, shifted)

    def timecrop(self, start, end):
        """Crop time response.

        Parameters
        ----------
        start, end : float
            Start and end times in seconds. Does not include sample at t=end. Use
            end=None to force inclusion of last sample.

        Returns
        -------
        Response
            New Response object with cropped time.

        Notes
        -----
        Creates new Response object.

        Examples
        --------
        >>> import numpy as np
        >>> from response import Response
        >>> r = Response.from_time(100, np.random.normal(size=100))
        >>> split = 0.2

        The following holds:

        >>> np.all(np.concatenate(
        ...     (
        ...         r.timecrop(0, split).in_time,
        ...         r.timecrop(split, None).in_time,
        ...     ),
        ...     axis=-1,
        ... ) == r.in_time)
        True

        """
        if start < 0:
            start += self.time_length
        if end is not None and end < 0:
            end += self.time_length
        assert 0 <= start < self.time_length
        assert end is None or (0 < end <= self.time_length)

        _, i_start = _find_nearest(self.times, start)
        if end is None:
            i_end = None
        else:
            _, i_end = _find_nearest(self.times, end)

        h = self.in_time[..., i_start:i_end]

        new_response = self.from_time(self.fs, h)

        return new_response

    def non_causal_timecrop(self, length):
        """Cut length of non-causal impulse response.

        "FFT shift, cropping on both ends, iFFT shift"

        Parameters
        ----------
        length : float
            final length in seconds

        Returns
        -------
        Response
            New Response object new length.

        Note
        ----
        Can introduce delay pre-delay by a sample.

        """
        assert length < self.time_length

        cut = (self.time_length - length) / 2

        _, i_start = _find_nearest(self.times, cut)
        _, i_end = _find_nearest(self.times, self.time_length - cut)

        h = np.fft.ifftshift(np.fft.fftshift(self.in_time)[..., i_start:i_end])

        new_response = self.from_time(self.fs, h)

        if new_response.time_length != length:
            w = f"Could not precisely shrink to {length}s with fs = {self.fs}"
            warnings.warn(w)

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
        """Zeropad time response to specific length.

        Returns
        -------
        Response
            New response object with new length n.

        """
        oldn = self.nt
        assert n >= oldn
        return self.zeropad(0, n - oldn)

    def resample(self, fs_new, normalize="same_gain", window=None):
        """Resample using Fourier method.

        Parameters
        ----------
        fs_new : int
            New sample rate
        normalize : str, optional
            If 'same_gain', normalize such that the gain is the same
            as the original signal. If 'same_amplitude', amplitudes will be preserved.
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
        if fs_new == self.fs:
            return self

        nt_new = fs_new * self.time_length

        if nt_new % 1 != 0:
            raise ValueError(
                "New number of samples must be integer, but is {}".format(nt_new)
            )

        nt_new = int(nt_new)

        h_new = resample(self.in_time, nt_new, axis=-1, window=window)

        if normalize == "same_gain":
            h_new *= self.nt / nt_new
        elif normalize == "same_amplitude":
            pass
        else:
            raise ValueError(
                "Expected 'same_gain' or 'same_amplitude', got %s" % (normalize,)
            )

        return self.from_time(fs_new, h_new)

    def resample_poly(self, fs_new, normalize="same_gain", window=("kaiser", 5.0)):
        """Resample using polyphase filtering.

        Parameters
        ----------
        fs_new : int
            New sample rate
        normalize : str, optional
            If 'same_gain', normalize such that the gain is the same
            as the original signal. If 'same_amplitude', amplitudes will be preserved.
        window : None, optional
            Passed to scipy.signal.resample_poly.

        Returns
        -------
        Response
            New resampled response object.

        """
        if fs_new == self.fs:
            return self

        ratio = Fraction(fs_new, self.fs)
        up = ratio.numerator
        down = ratio.denominator

        if up > 1000 or down > 1000:
            print("Warning: resampling with high ratio {}/{}".format(up, down))

        h_new = resample_poly(self.in_time, up, down, axis=-1, window=window)

        if normalize == "same_gain":
            h_new *= down / up
        elif normalize == "same_amplitude":
            pass
        else:
            raise ValueError(
                "Expected 'same_gain' or 'same_amplitude', got %s" % (normalize,)
            )

        return self.from_time(fs_new, h_new)

    def normalize(self, maxval=1):
        """Normalize time response.

        Parameters
        ----------
        maxval: float, optional
            Maximum amplitude in resulting time response.

        Returns
        -------
        Response

        """
        h = self.in_time
        h /= np.abs(self.in_time).max()
        h *= maxval
        return self.from_time(self.fs, h)

    def export_wav(self, folder, name_fmt="{:02d}.wav", dtype=np.int16):
        """Export impulse response to wave files.

        Dimension of data must 2.

        Parameters
        ----------
        folder : file path
            Save in this folder
        name_fmt : str, optional
            Format string for file names with one placeholder, e.g. 'filt1{:02d}.wav'.
        dtype : one of np.float32, np.int32, np.int16, np.uint8
            Data is converted to this type.

        """
        data = np.atleast_2d(self.in_time)

        assert data.ndim == 2
        assert np.all(np.abs(data) <= 1.0)

        # convert and scale to new output datatype
        if dtype in [np.uint8, np.int16, np.int32]:
            lim_orig = (-1.0, 1.0)
            lim_new = (np.iinfo(dtype).min, np.iinfo(dtype).max)
            data = _rescale(data, lim_orig, lim_new).astype(dtype)
        elif dtype != np.float32:
            raise TypeError(f"dtype {dtype} is not supported by scipy.wavfile.write.")

        path = Path(folder)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=False)

        for i in range(data.shape[0]):
            wavfile.write(path / name_fmt.format(i + 1), self.fs, data[i])

    def export_npz(self, filename, dtype=np.float32):
        """Export impulse response as npz file.

        Parameters
        ----------
        filename: str or Path
            File path
        dtype: numpy dtype
            Convert to this type before saving

        """
        np.savez(
            filename, impulse_response=self.in_time.astype(dtype), samplerate=self.fs
        )

    def power_in_bands(self, bands=None, avgaxis=None):
        """Compute power of signal in third octave bands.

        Power(band) =   1/T  integral  |X(f)| ** 2 df
                            f in band

        Parameters
        ----------
        bands : list of tuples, length nbands optional
            Center, lower and upper frequencies of bands.
        avgaxis: int, tuple or None
            Average result over these axis

        Returns
        -------
        P: ndarray, shape (..., nbands)
            Power in bands
        fcs: list, length nbands
            Center frequencies of bands

        """
        if bands is None:
            bands = _third_octave_bands

        # center frequencies
        fcs = np.asarray([b[0] for b in bands])
        Npow2 = 2 ** (self.nt - 1).bit_length()
        f = np.fft.fftfreq(Npow2, d=1 / self.fs)

        shape = list(self.in_freq.shape)
        shape[-1] = len(bands)
        P = np.zeros(shape)
        for i, (fc, fl, fu) in enumerate(bands):
            if fu < self.fs / 2:  # include only bands in frequency range
                iband = np.logical_and(fl <= f, f < fu)
                P[..., i] = np.sum(
                    np.abs(np.fft.fft(self.in_time, n=Npow2, axis=-1)[..., iband]) ** 2
                    * 2  # energy from negative and positive frequencies
                    * self.dt
                    / self.nt
                    / self.time_length,
                    axis=-1,
                )
            else:
                P[..., i] = 0

        if avgaxis is not None:
            P = P.mean(axis=avgaxis)

        return P, fcs

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
    def freq_vector(cls, n, fs):
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
        return freq_vector(n, fs, sided='single')

    def filter(self, b, a=[1]):
        """Filter response along one-dimension with an IIR or FIR filter.

        Parameters
        ----------
        b : array_like
            The numerator coefficient vector in a 1-D sequence.
        a : array_like, optional
            The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
            is not 1, then both `a` and `b` are normalized by ``a[0]``.

        """
        return self.from_time(self.fs, lfilter(b, a, self.in_time, axis=-1))

    def add_noise(self, snr, unit=None):
        """Add noise to x with relative noise level SNR.

        Parameters
        ----------
        snr : float
            relative magnitude of noise, i.e. snr = Ex/En
        unit : None or str, optional
            if "dB", SNR is specified in dB, i.e. SNR = 10*log(Ex/En).

        Returns
        -------
        Response

        """
        return self.from_time(self.fs, noisify(self.in_time, snr, unit=unit))

    def psd(self, **kwargs):
        """Compute the power spectral density of the signal.

        Parameters
        ----------
        kwargs
            keword arguments passed to scipy.signal.welch

        Returns
        -------
        f : ndarray
            Array of sample frequencies.
        Pxx : ndarray
            Power spectral density of time signal.

        Notes
        -----
        Use scaling='density' for power per bin bandwidth and scaling='spectrum' for
        power per bin.

        """
        return welch(self.in_time, fs=self.fs, **kwargs)


####################
# Module functions #
####################


def noisify(x, snr, unit=None):
    """Add noise to x with relative noise level SNR.

    Parameters
    ----------
    x : ndarray
        data
    snr : float
        relative energy of noise, snr = Energy(x)/Energy(n).
    unit : None or str, optional
        if "dB", snr is specified in dB, i.e. snr = 10*log(Ex/En).

    Returns
    -------
    ndarray
        data with noise

    Examples
    --------
    Create signal

    >>> import numpy as np
    >>> t = np.linspace(0, 1, 1000000, endpoint=False)
    >>> x = np.sin(2*np.pi*10*t)  # signal

    Add noise with 6 dB SNR to a sinusoidal signal:

    >>> snrdB = 6
    >>> xn = noisify(x, snrdB, "dB")  # signal plus noise

    >>> energy_x = np.linalg.norm(x)**2
    >>> energy_xn = np.linalg.norm(xn)**2
    >>> snr = 10 ** (snrdB / 20)
    >>> np.allclose((1 + 1/snr) * energy_x, energy_xn, rtol=1e-2)
    True

    """
    if unit == "dB":
        snr = 10 ** (snr / 20)

    if np.iscomplexobj(x):
        n = np.random.standard_normal(x.shape) + 1j * np.random.standard_normal(x.shape)
    else:
        n = np.random.standard_normal(x.shape)

    n *= 1 / np.sqrt(snr) * np.linalg.norm(x) / np.linalg.norm(n)

    return x + n


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
    sided: str
        Generate frequencies for a "single" or "double" sided spectrum

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


def delay(fs, x, dt, keep_length=True, axis=-1):
    """Delay time signal by dt seconds by inserting zeros.

    Examples
    --------
    >>> delay(1, [1, 2, 3], 1)
    array([0., 1., 2.])

    >>> delay(1, [1, 2, 3], 1, keep_length=False)
    array([0., 1., 2., 3.])

    >>> delay(1, [1, 0, 0], -1)
    array([0., 0., 0.])

    >>> delay(1, [1, 0, 0], -1, keep_length=False)
    array([0, 0])

    """
    dn = int(round(dt * fs))
    x = np.asarray(x)
    n = x.shape[axis]

    if dn > 0:
        # delay
        zeros_shape = list(x.shape)
        zeros_shape[axis] = dn
        zeros = np.zeros(zeros_shape)

        delayed = np.concatenate((zeros, x), axis=axis)

        if keep_length:
            # slice that takes 0 to ntaps samples along axis
            slc = [slice(None)] * len(x.shape)
            slc[axis] = slice(0, n)
            delayed = delayed[tuple(slc)]

    elif dn < 0:
        # pre-delay
        slc = [slice(None)] * len(x.shape)
        slc[axis] = slice(-dn, n)
        delayed = x[tuple(slc)]

        if keep_length:
            zeros_shape = list(x.shape)
            zeros_shape[axis] = -dn
            zeros = np.zeros(zeros_shape)
            delayed = np.concatenate((delayed, zeros), axis=axis)
    else:
        # no delay
        delayed = x

    return delayed


def delay_between(h1, h2):
    """Estimate delay of h2 relative to h1 using cross correlation.

    Parameters
    ----------
    h1 : ((N,) L) array_like
        Reference signals.
    h2 : ((M,) L) array_like
        Delayed signals.

    Returns
    -------
    delay : (N, M) ndarray
        Delays in samples. `h2[j]` is delayed relative to `h1[i]` by `delay[i, j]`.

    Examples
    --------
    >>> a = [1, 0, 0, 0]
    >>> b = [0, 0, 1, 0]
    >>> delay_between(a, b)
    array(2)

    """
    h1 = np.atleast_2d(h1)
    h2 = np.atleast_2d(h2)
    assert h1.shape[-1] == h2.shape[-1], "h1 and h2 must have same number of samples"

    L = h1.shape[-1]

    delay = np.zeros((h1.shape[0], h2.shape[0]), dtype=int)
    for i in range(h1.shape[0]):
        for j in range(h2.shape[0]):
            xcorrmax = np.argmax(np.correlate(h2[j], h1[i], mode="full"))
            delay[i, j] = xcorrmax - L + 1

    return delay.squeeze()


def align(h, href, upsample=1):
    """Align two impulse responses using cross correlation.

    Parameters
    ----------
    h : array_like
        Response that will be aligned.
    href : array_like
        Response to which will be aligned.
    upsample : int, optional
        Upsample both responses before alignment by this factor.

    Returns
    -------
    ndarray
        Time aligned version of `h`.

    """
    href = resample_poly(href, upsample, 1)
    h = resample_poly(h, upsample, 1)
    delay = delay_between(href, h).squeeze()
    h = np.roll(h, -int(delay))
    h = resample_poly(h, 1, upsample)
    return h


#####################
# Utility functions #
#####################


def _sample_window(n, startwindow, stopwindow, window="hann"):
    """Create a sample domain window."""
    swindow = np.ones(n)

    if startwindow is not None:
        length = startwindow[1] - startwindow[0]
        w = get_window(window, 2 * length, fftbins=False)[:length]
        swindow[: startwindow[0]] = 0
        swindow[startwindow[0] : startwindow[1]] = w

    if stopwindow is not None:
        # stop window
        length = stopwindow[1] - stopwindow[0]
        w = get_window(window, 2 * length, fftbins=False)[length:]
        swindow[stopwindow[0] + 1 : stopwindow[1] + 1] = w
        swindow[stopwindow[1] + 1 :] = 0

    return swindow


def _time_window(fs, n, startwindow_t, stopwindow_t, window="hann"):
    """Create a time domain window.

    Negative times are relative to the end. Short cut for end time is `None`.
    """
    times = time_vector(n, fs)
    T = times[-1] + times[1]  # total time length

    if startwindow_t is None:
        startwindow_n = None
    else:
        startwindow_n = []
        for t in startwindow_t:
            if t < 0:
                t += T
            assert 0 <= t or t <= T
            startwindow_n.append(_find_nearest(times, t)[1])

    if stopwindow_t is None:
        stopwindow_n = None
    else:
        stopwindow_n = []
        for t in stopwindow_t:
            if t is None:
                t = times[-1]
            elif t < 0:
                t += T
            assert 0 <= t or t <= T
            stopwindow_n.append(_find_nearest(times, t)[1])

    twindow = _sample_window(n, startwindow_n, stopwindow_n, window=window)

    return twindow


def _freq_window(fs, n, startwindow_f, stopwindow_f, window="hann"):
    """Create a frequency domain window."""
    freqs = freq_vector(n, fs)

    if startwindow_f is not None:
        startwindow_n = [_find_nearest(freqs, f)[1] for f in startwindow_f]
    else:
        startwindow_n = None

    if stopwindow_f is not None:
        stopwindow_n = [_find_nearest(freqs, f)[1] for f in stopwindow_f]
    else:
        stopwindow_n = None

    fwindow = _sample_window(len(freqs), startwindow_n, stopwindow_n, window=window)

    return fwindow


def _rescale(x, xlim, ylim):
    """Rescale values to new bounds.

    Parameters
    ----------
    x : ndarray
        Values to rescale
    xlim : tuple
        Original value bounds (xmin, xmax)
    ylim : float
        New value bounds (ymin, ymax)

    Returns
    -------
    ndarray
        Rescaled values

    """
    m = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    c = ylim[1] - m * xlim[1]
    y = m * x + c
    return y


def _find_nearest(array, value):
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


def _construct_window_around_peak(fs, irs, tleft, tright, alpha=0.5):
    """Create time window around maximum of response.

    Parameters
    ----------
    fs : int
        Sample rate.
    irs : array_like
        Input response.
    tleft : float
        Start of time window relative to impulse response peak.
    tright : float
        End of time window relative to impulse response peak.
    alpha : float, optional
        Alpha parameter of Tukey window.

    Returns
    -------
    ndarray
        Time windows.

    """
    orig_shape = irs.shape
    flat_irs = irs.reshape(-1, irs.shape[-1])

    sleft = int(fs * tleft)
    sright = int(fs * tright)

    windows = np.ones(flat_irs.shape)
    for i in range(flat_irs.shape[0]):
        ipeak = np.argmax(np.abs(flat_irs[i]))
        iwstart = max(ipeak - sleft, 0)
        iwend = min(ipeak + sright, flat_irs.shape[-1])

        window = tukey(iwend - iwstart, alpha=alpha)

        windows[i, iwstart:iwend] *= window
        windows[i, :iwstart] = 0
        windows[i, iwend:] = 0

    return windows.reshape(orig_shape)


def _aroll(x, n, circular=False, axis=-1, copy=True):
    """Roll each entry along axis individually.

    Can be used to delay / shift each response by its own shift.

    Parameters
    ----------
    x : ndarray (Ni...,M,Nj...)
        Input array
    n : ndarray (Ni...,Nj...)
        Delay times of each entry along axis.
    circular: bool, optional
        If True, wrap around ends. Else replace with zeros.
    axis : int, optional
        Axis along which is rolled.
    copy : bool, optional
        If True, operate on copy of `x`. Else roll inplace.

    Returns
    -------
    ndarray (Ni...,M,Nj...)
        Array with rolled entries

    """
    n = n.astype(int)

    if copy:
        x = x.copy()

    # move axis to first dim and reshape to 2D
    xview = np.rollaxis(x, axis)
    xview = xview.reshape(xview.shape[0], -1)
    n = n.reshape(-1)

    assert n.shape[0] == xview.shape[1], 'Shapes of x and n do not match.'

    for i in range(n.shape[0]):
        xview[:, i] = np.roll(xview[:, i], n[i])

        if not circular:
            if n[i] > 0:
                xview[: n[i], i] = 0
            elif n[i] < 0:
                xview[n[i] :, i] = 0

    return x


# center, lower, upper frequency of third octave bands
_third_octave_bands = (
    (16, 13.920_292_470_942_801, 17.538_469_504_833_955),
    (20, 17.538_469_504_833_95, 22.097_086_912_079_607),
    (25, 22.097_086_912_079_615, 27.840_584_941_885_613),
    (31, 27.840_584_941_885_602, 35.076_939_009_667_91),
    (40, 35.076_939_009_667_9, 44.194_173_824_159_215),
    (50, 44.194_173_824_159_23, 55.681_169_883_771_226),
    (63, 55.681_169_883_771_204, 70.153_878_019_335_82),
    (80, 70.153_878_019_335_82, 88.388_347_648_318_44),
    (100, 88.388_347_648_318_43, 111.362_339_767_542_41),
    (125, 111.362_339_767_542_41, 140.307_756_038_671_64),
    (160, 140.307_756_038_671_64, 176.776_695_296_636_9),
    (200, 176.776_695_296_636_86, 222.724_679_535_084_82),
    (250, 222.724_679_535_084_82, 280.615_512_077_343_3),
    (315, 280.615_512_077_343_3, 353.553_390_593_273_8),
    (400, 353.553_390_593_273_8, 445.449_359_070_169_75),
    (500, 445.449_359_070_169_63, 561.231_024_154_686_6),
    (630, 561.231_024_154_686_6, 707.106_781_186_547_6),
    (800, 707.106_781_186_547_6, 890.898_718_140_339_5),
    (1000, 890.898_718_140_339_3, 1122.462_048_309_373_1),
    (1260, 1122.462_048_309_373_1, 1414.213_562_373_095),
    (1600, 1414.213_562_373_094_9, 1781.797_436_280_678_5),
    (2000, 1781.797_436_280_678_5, 2244.924_096_618_746_3),
    (2500, 2244.924_096_618_746_3, 2828.427_124_746_19),
    (3200, 2828.427_124_746_19, 3563.594_872_561_358),
    (4000, 3563.594_872_561_357, 4489.848_193_237_492_5),
    (5000, 4489.848_193_237_492_5, 5656.854_249_492_38),
    (6300, 5656.854_249_492_379_5, 7127.189_745_122_714),
    (8000, 7127.189_745_122_714, 8979.696_386_474_985),
    (10000, 8979.696_386_474_985, 11313.708_498_984_76),
    (12600, 11313.708_498_984_759, 14254.379_490_245_428),
    (16000, 14254.379_490_245_428, 17959.392_772_949_97),
    (20000, 17959.392_772_949_966, 22627.416_997_969_518),
)


def _add_octave_band_xticks(ax, bands=np.array(_third_octave_bands)[:, 0]):
    """Add band ticks to axis."""
    left, right = ax.get_xlim()
    b = bands[np.logical_and(left <= bands, bands <= right)]
    ax.set_xticks(b)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([int(round(bs)) for bs in b], minor=False)
