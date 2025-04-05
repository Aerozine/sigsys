.PHONY: prun jrun clean 
run : prun jrun
init:
	julia -e 'using Pkg;Pkg.instantiate()'
jrun:
	julia --threads=auto  main.jl
jformat:
	julia -e 'using JuliaFormatter; format(".", SciMLStyle())'

prun:
	python main.py	

clean:

