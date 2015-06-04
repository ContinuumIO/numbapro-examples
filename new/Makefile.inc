
%.empty.ipynb: %.py
	python -m py2nb $< $@


%.ipynb: %.empty.ipynb
	ipython nbconvert --to=notebook --execute --output=$@ $<


%.html: %.ipynb
	ipython nbconvert --to=html --output=$@ $<
