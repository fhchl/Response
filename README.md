Response
=============================

## Testing

Run tests in base directory with

	nosetests

Always run tests before pushing code.

## Developing

Install for development with `pip install -e .` .

Comments should comply with the [Numpy/Scipy documentation style][1]. An
example can also be found [here][2].

Code should comply to the [pep8 coding style][3]. You can check if the code complies
with

	find . -name \*.py -exec pep8 {} +


[1]: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
[2]: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
[3]: https://www.python.org/dev/peps/pep-0008/
