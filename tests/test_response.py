# -*- coding: utf-8 -*-

from .context import response  # noqa

from response import Response

import unittest
import numpy as np
import numpy.testing as npt


class TestResponse(unittest.TestCase):
    def test_has_fs(self):
        fs = 64
        nf = 64
        n_s = 2
        n_mic = 3
        fdata = np.ones((n_s, n_mic, nf))
        resp = Response.from_freq(fs, fdata)
        self.assertEqual(fs, resp.fs)

    def test_has_frequency_data(self):
        fs = 64
        nf = 64
        n_s = 2
        n_mic = 3
        fdata = np.ones((n_s, n_mic, nf))
        resp = Response.from_freq(fs, fdata)
        npt.assert_equal(fdata, resp.in_frequency)

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
        npt.assert_almost_equal(resp.in_frequency, 1)

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


if __name__ == '__main__':
    unittest.main()
