all: build/V46.pdf

 #hier Python-Skripte:
build/plot1.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py 
build/plot2.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py 
build/plot3.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py 
build/plot4.pdf: python python/plot.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/plot.py 

 # hier weitere Abhängigkeiten für build/main.pdf deklarieren:
 build/V46.pdf: build/plot1.pdf \
				build/plot2.pdf \
				build/plot3.pdf \
				build/plot4.pdf \
 			   
build/V46.pdf: FORCE | build
	  TEXINPUTS=build: \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	V46.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
