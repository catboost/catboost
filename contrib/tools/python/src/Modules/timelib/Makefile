all:: timelib.c
	python setup.py build

clean::
	rm -rf *.so build/ dist/

install:: timelib.c
	python setup.py build install

sdist:: timelib.c
	python setup.py build sdist

ext-date-lib/parse_date.c: ext-date-lib/parse_date.re
	re2c -d -b -o ext-date-lib/parse_date.c ext-date-lib/parse_date.re

%.c: %.pyx
	cython $< -o $@
