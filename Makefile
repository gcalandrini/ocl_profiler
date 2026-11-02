
RM := rm 
#-$(RM) Release/libocl_profiler.so Debug/libocl_profiler.so

all:
	make release

debug:
	@echo Building debug configuration
	make -C Debug -f makefile

release:
	@echo Building release configuration
	make -C Release -f makefile

clean:
	make clean -C Release -f makefile 
	make clean -C Debug -f makefile
	@echo 'Clean done'