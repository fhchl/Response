# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as npt
import response
from response import Response

from .context import response  # noqa


class TestCreation:
    def test_has_fs(self):
        fs = 64
        nf = 64
        n_s = 2
        n_mic = 3
        fdata = np.ones((n_s, n_mic, nf))
        resp = Response.from_freq(fs, fdata)
        assert fs == resp.fs

    def test_has_frequency_data(self):
        fs = 64
        nf = 64
        n_s = 2
        n_mic = 3
        fdata = np.ones((n_s, n_mic, nf))
        resp = Response.from_freq(fs, fdata)
        npt.assert_equal(fdata, resp.in_freq)

    def test_flat_converts_to_dirac(self):
        fs = 64
        nt = 64
        nf = nt // 2 + 1
        n_s = 2
        n_mic = 3
        fdata = np.ones((n_s, n_mic, nf))
        resp = Response.from_freq(fs, fdata)

        npt.assert_almost_equal(0, resp.in_time[:, :, 1:])
        npt.assert_almost_equal(1, resp.in_time[:, :, 0])

    def test_dirac_converts_to_flat(self):
        fs = 64
        nt = 64
        n_s = 2
        n_mic = 3
        tdata = np.zeros((n_s, n_mic, nt))
        tdata[:, :, 0] = 1
        resp = Response.from_time(fs, tdata)
        npt.assert_almost_equal(resp.in_freq, 1)


class TestMethods:
    def test_timecrop_splits_concatenate_properly(self):
        fs = 100
        T = 1.12
        t = np.linspace(0, T, int(T * fs), endpoint=False)
        x = np.sin(t * np.pi * 2 * 10)

        split = 0.5

        assert np.all(
            np.concatenate(
                (
                    Response.from_time(fs, x).timecrop(0, split).in_time,
                    Response.from_time(fs, x).timecrop(split, None).in_time,
                ),
                axis=-1,
            )
            == x
        )

        split = 0.1

        assert np.all(
            np.concatenate(
                (
                    Response.from_time(fs, x).timecrop(0, split).in_time,
                    Response.from_time(fs, x).timecrop(split, None).in_time,
                ),
                axis=-1,
            )
            == x
        )

        split = 0.611421251

        assert np.all(
            np.concatenate(
                (
                    Response.from_time(fs, x).timecrop(0, split).in_time,
                    Response.from_time(fs, x).timecrop(split, None).in_time,
                ),
                axis=-1,
            )
            == x
        )

    def test_pad_to_power_of_2(self):
        fs = 64
        nt = 204
        n_s = 2
        n_mic = 3
        tdata = np.empty((n_s, n_mic, nt))
        r = Response.from_time(fs, tdata)
        rpad = r.zeropad_to_power_of_2()

        assert rpad.nt == 256
        npt.assert_equal(r.in_time, rpad.in_time[..., :nt])


class TestFunctions:
    def test_aroll(self):
        x = np.eye(10)
        dn = np.arange(10)

        rolled = response._aroll(x, -dn)

        assert np.all(rolled[:, :1] == 1)
        assert np.all(rolled[:, 1:] == 0)

    def test_non_causal_set_to_length(self):
        h = [1, 0, 0, 1]
        r = Response.from_time(1, h)

        npt.assert_equal(r.non_causal_set_to_length(6).in_time, [1, 0, 0, 0, 0, 1])
        npt.assert_equal(r.non_causal_set_to_length(2).in_time, [1, 1])
