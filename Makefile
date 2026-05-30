PYTHON ?= python

all:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build *.so *.dylib *.pyd __pycache__
