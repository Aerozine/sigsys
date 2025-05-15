.PHONY: jrun clean 
run : jrun
init:
	julia --project=. -e 'using Pkg;Pkg.instantiate()'
jrun:
	julia --project=. --threads=auto main.jl
jrunfast:
	julia --project=. --optimize=0 --compile=min --threads=auto --math-mode=fast  main.jl
jformat:
	julia -e 'using JuliaFormatter; format(".", SciMLStyle())'
clean:
	rm jlplots/*.pdf best*.txt

