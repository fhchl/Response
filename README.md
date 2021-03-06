Response
========

_Your handy frequency and impulse response processing object!_

[![](https://img.shields.io/pypi/l/response.svg?style=flat)](https://pypi.org/project/response/)
[![](https://img.shields.io/pypi/v/response.svg?style=flat)](https://pypi.org/project/response/)
[![travis-ci](https://travis-ci.org/fhchl/Response.svg?branch=master)](https://travis-ci.org/fhchl/Response)
[![codecov](https://codecov.io/gh/fhchl/Response/branch/master/graph/badge.svg)](https://codecov.io/gh/fhchl/Response)

This module supplies the `Response` class: an abstraction of frequency and
impulse responses and a set of handy methods for their processing. It implements a
[fluent interface][1] for chaining the processing commands.

Find the documentation [here][2] and the source code on [GitHub][3].

[1]: https://en.wikipedia.org/wiki/Fluent_interface
[2]: https://fhchl.github.io/Response/
[3]: https://github.com/fhchl/Response

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
