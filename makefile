
.PHONY:all run code rapport 
all: code rapport
run: code rapport

code:
	make -C code jrun
rapport:
	make -C rapport build

clean:
	make clean -C code 
	make clean -C rapport 
