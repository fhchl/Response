Response
========

(work in progress)

The response module defines the `Response` class as an abstraction of frequency and impulse responses.

```python
from response import Response
fs = 16
x = [1]
# Create impulsive response and do chain of processing
r = Response.from_time(fs, x) \
            .zeropad(0, 15) \
            .delay(0.5) \
            .resample(10 * fs, window=('kaiser', 0.5)) \
            .timecrop(0, 0.6) \
            .time_window((0, 0.2), (0.5, 0.6))
# plot result
r.plot(show=True)
# time domain data
r.in_time
# frequency domain data
r.in_freq
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


[1]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
[2]: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
[3]: https://www.python.org/dev/peps/pep-0008/
