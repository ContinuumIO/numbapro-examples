include Makefile.inc

PROJECTS += matrixmult_intro
PROJECTS += matrixmult_sharedmemory

PYTHONSCRIPTS = $(addsuffix .py, $(PROJECTS))
EMPTYNOTEBOOKS = $(addsuffix .empty.ipynb, $(PROJECTS))
NOTEBOOKS = $(addsuffix .ipynb, $(PROJECTS))

TARGETS=$(NOTEBOOKS)

all: $(TARGETS)

clean:
	rm -f $(EMPTYNOTEBOOKS) $(NOTEBOOKS)
