# Using `long double` instead of `double` for improved precision
LONGDOUBLE = 0

# Use CUDA/GPU to perform the calculation of acceleratoins
GPU = 0

# Use OpenCL/GPU to perform the calculation of accelerations
OPENCL = 0
# Stringify the OpenCL kernel 
LOAD_KERNEL_FROM_STRING = 1

# Accelerate using OPENMP
OPENMP = 1

# CUDA installation path
CUDA_SDK_PATH=/usr/local/cuda

# OpenCL installation path
OPENCL_SDK_PATH=/opt/opencl

# path to Python interpreter
PYTHON = python3

CFLAGS += -fPIC -O3 -std=c99 -g -fcommon -fstrict-aliasing -shared

OBJS = common.o integrator_runge_kutta.o integrator_gauss_radau15.o integrator_wisdom_holman.o additional_forces.o

DEPS = common.h

ifeq ($(GPU), 1)
	OBJS += gpuforce.o 
	CFLAGS += -DGPU -L${CUDA_SDK_PATH}/lib64 -I${CUDA_SDK_PATH}/include -lcudart -lstdc++
else
ifeq ($(OPENCL), 1)
	OBJS += openclforce.o
	CFLAGS += -DOPENCL -I${OPENCL_SDK_PATH}/include -L${OPENCL_SDK_PATH}/lib -lOpenCL
ifeq ($(LOAD_KERNEL_FROM_STRING), 1)
	CFLAGS += -DLOAD_KERNEL_FROM_STRING
endif
endif
endif

ifeq ($(OPENMP), 1)
	CFLAGS += -DOPENMP -fopenmp
endif

ifeq ($(LONGDOUBLE), 1)
	CFLAGS += -DLONGDOUBLE
endif

ifeq ($(GPU), 1)
	NVCC = nvcc
else
	NVCC = $(CC)
endif

# Determine whether on minGW
ifeq ($(OS),Windows_NT)
	FINALTARGET = libabie.dll
	CFLAGS += -DWIN32 -DLIBABIE_EXPORTS -m64
else
	FINALTARGET = libabie.so
endif

libabie.so: $(OBJS)
ifeq ($(GPU), 1)
	$(CC) -o $@ $(OBJS) $(CFLAGS)
	mv libabie.so ../ABIE/$(FINALTARGET)
	
else
	$(CC) -o $@ $(OBJS) $(CFLAGS)
	mv libabie.so ../ABIE/$(FINALTARGET)
endif

gpuforce.o: gpuforce.cu
ifeq ($(GPU), 1)
	$(NVCC) -Xcompiler -fPIC -DGPU -O3 -g $^ -c -o $@
endif

openclforce.o: openclforce.c $(DEPS) force_kernel.cl
ifeq ($(OPENCL), 1)
ifeq ($(LOAD_KERNEL_FROM_STRING), 1)
	$(PYTHON) ../source2string.py -i force_kernel.cl kernel_string.h
else
	cp force_kernel.cl ../examples/force_kernel.cl
endif
	$(CC) -c -o $@ $< $(CFLAGS)
endif


%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f *.o kernel_string.h ../ABIE/$(FINALTARGET)
