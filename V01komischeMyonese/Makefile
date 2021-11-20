all: build/V01.pdf

#hier Python-Skripte:
build/plot1.pdf: python/test.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/test.py 
build/plot11.pdf: python/test2.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/test2.py 
build/plot12.pdf: python/test2.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python python/test2.py 
	 
# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
build/V01.pdf: build/plot1.pdf \
			   build/plot11.pdf \
			   build/plot12.pdf \

build/V01.pdf: FORCE | build
	  TEXINPUTS=build: \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	V01.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
