.PHONY: jrun clean 
run : jrun
init:
	julia -e 'using Pkg;Pkg.instantiate()'
jrun:
	julia --threads=auto  main.jl
jrunfast:
	julia --optimize=0 --compile=min --threads=auto  main.jl
jformat:
	julia -e 'using JuliaFormatter; format(".", SciMLStyle())'
clean:
	rm jlplots/*.pdf

