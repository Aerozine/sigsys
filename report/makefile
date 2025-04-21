TEX_FILE = clean.tex
PDF_FILE = clean.pdf

# Default target
all: $(PDF_FILE)
build: $(PDF_FILE)


$(PDF_FILE): $(TEX_FILE) 
	latexmk -pdf -xelatex $(TEX_FILE)
	latexmk -c

clean:
	latexmk -C

.PHONY: all clean build
