
.PHONY:all run code report 
all: code report
run: code report

NPROCS = $(shell nprocs)
code:
	$(MAKE) -j$(NPROCS) -C code init 
	$(MAKE) -j$(NPROCS) -C code jrun
report: code
	$(MAKE) -j$(NPROCS) -C report build
latex: 
	$(MAKE) -j$(NPROCS) -C build
clean:
	make clean -C code 
	make clean -C report 
