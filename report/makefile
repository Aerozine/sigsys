# Makefile to compile clean.tex using xelatex with parallelism
TEX_FILE = clean.tex
PDF_FILE = clean.pdf
JL_PLOTS_DIR = ../code/jlplots

# Default target
all: $(PDF_FILE)
build: $(PDF_FILE)


$(PDF_FILE): $(TEX_FILE) $(JL_PLOTS_DIR)/*
	cp -R ../code/jlplots .
	latexmk -pdf -xelatex $(TEX_FILE)
	rm -rf jlplots
	latexmk -c

clean:
	latexmk -C

.PHONY: all clean build
