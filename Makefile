CC      ?= cc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall -Wextra
# make      -> CPU (OpenMP) build
# make gpu  -> adds the OpenCL backend (needs OpenCL headers + ICD loader)
miso: miso.c
	$(CC) $(CFLAGS) miso.c -o miso -lm
gpu: miso.c
	$(CC) $(CFLAGS) -DUSE_OPENCL miso.c -o miso -lOpenCL -lm
# macOS: cc -O3 -DUSE_OPENCL miso.c -o miso -framework OpenCL -lm   (Apple clang: no -fopenmp; use brew libomp)
# Windows (MSVC): cl /O2 /openmp /DUSE_OPENCL miso.c OpenCL.lib
clean:
	rm -f miso

.PHONY: gpu clean
