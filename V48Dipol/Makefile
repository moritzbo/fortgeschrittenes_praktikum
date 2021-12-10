all: build/V48.pdf

#hier Python-Skripte:
build/Temperatur.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py 
build/Strom1.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py 
build/Strom2.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py 
build/untegrrund1.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py 
build/untegrrund2.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/V48.pdf: build/Temperatur.pdf \
			   build/Strom1.pdf \
			   build/Strom2.pdf \
			   build/untegrrund1.pdf \
			   build/untegrrund2.pdf \

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
