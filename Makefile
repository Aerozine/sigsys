
.PHONY: proj1 proj2 proj3 rapport 

proj1:
	make -C proj1 run
proj2:
	make -C proj2 run
proj3:
	make -C proj3 run
rapport:
	make -C rapport build

clean:
	make clean -C proj1
	make clean -C proj2
	make clean -C proj3
	make clean -C rapport 
