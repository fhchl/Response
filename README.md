Response
========

[![](https://img.shields.io/pypi/l/response.svg?style=flat)](https://pypi.org/project/response/)
[![](https://img.shields.io/pypi/v/response.svg?style=flat)](https://pypi.org/project/response/)
[![travis-ci](https://travis-ci.org/fhchl/Response.svg?branch=master)](https://travis-ci.org/fhchl/Response)
[![codecov](https://codecov.io/gh/fhchl/Response/branch/master/graph/badge.svg)](https://codecov.io/gh/fhchl/Response)

The response module defines the `Response` class as an abstraction of frequency and impulse responses.

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
# plot result
r.plot(show=True)
# real impulse response
r.in_time
# complex frequency response
r.in_freq
# and much more ...
```

## Testing

Run tests in base directory with

	pytest

Always run tests before pushing code.

## Developing

Install for development with `pip install -e .` .

Comments should comply with the [Numpy/Scipy documentation style][1]. An
example can also be found [here][2]. Code should comply to the [pep8 coding style][3]. You can check if the code complies by executing

    pycodestyle
    pydocstyle

Git commit messages follow [Angular style][4]. Uses [python-semantic-release][5] for versioning on travis-ci for pushs to master branch.

[1]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
[2]: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
[3]: https://www.python.org/dev/peps/pep-0008/
[4]: https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines
[5]: https://github.com/relekang/python-semantic-release
