include Makefile.inc

# Make new project names here
PROJECTS += matrixmult_intro
PROJECTS += matrixmult_sharedmemory


# Don't modify the rest of the file
PYTHONSCRIPTS = $(addsuffix .py, $(PROJECTS))
EMPTYNOTEBOOKS = $(addsuffix .empty.ipynb, $(PROJECTS))
NOTEBOOKS = $(addsuffix .ipynb, $(PROJECTS))
HTMLS = $(addsuffix .html, $(PROJECTS))

TARGETS=$(NOTEBOOKS)

all: $(TARGETS)

html: $(HTMLS)

clean:
	rm -f $(EMPTYNOTEBOOKS) $(NOTEBOOKS) $(HTMLS)

clean-html:
	rm -f $(HTMLS)

clean-empty:
	rm -f $(EMPTYNOTEBOOKS)

clean-push: clean-html clean-empty
