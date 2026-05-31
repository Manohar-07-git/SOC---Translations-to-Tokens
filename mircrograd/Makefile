# Makefile to clean LaTeX temporary files (only in current directory)

# List of LaTeX temp file extensions to delete
TEMP_EXTENSIONS = aux bbl blg log  out synctex.gz fdb_latexmk xml toc 

.PHONY: clean

.RECIPEPREFIX = >
all:
>pdflatex report.tex
>pdflatex report.tex # Run twice for table of contents
cleantemp:
>rm -f *.aux *.log *.out *.toc # to delete all created files except the pdf file
clean:
>rm -f *.aux *.log *.out *.toc *.pdf # to delete all files created including the output pdf file