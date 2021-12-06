all: build/V48.pdf

#hier Python-Skripte:
#build/plot1.pdf: python/test.py matplotlibrc header-matplotlib.tex | build
#	TEXINPUTS=$$(pwd): python python/test.py 

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
#build/V48.pdf:

build/V48.pdf: FORCE | build
	  TEXINPUTS=build: \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	V48.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
