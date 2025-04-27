.PHONY: jrun clean 
run : jrun
init:
	julia -e 'using Pkg;Pkg.instantiate()'
jrun:
	julia --threads=auto --math-mode=ieee main.jl
jrunfast:
	julia --optimize=0 --compile=min --threads=auto --math-mode=fast  main.jl
jformat:
	julia -e 'using JuliaFormatter; format(".", SciMLStyle())'
clean:
	rm jlplots/*.pdf

