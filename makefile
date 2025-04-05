
.PHONY:all run code rapport 
all: code rapport
run: code rapport

NPROCS = $(shell nprocs)
code:
	$(MAKE) -j$(NPROCS) -C code jrun
rapport:
	$(MAKE) -j$(NPROCS) -C rapport build

clean:
	make clean -C code 
	make clean -C rapport 
