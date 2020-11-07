#ifdef OPENCL

#define USE_SHARED 1
#define LOAD_FROM_STRING

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef LOAD_FROM_STRING
#include "kernel_string.h"
#endif

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl_ext.h>
#include "common.h"

#define MAX_SOURCE_SIZE 0x10000
#define MAX_PLATFORMS 4

static cl_mem pos_dev = NULL;
static cl_mem acc_dev = NULL;
static double* pos_host = NULL;
static double* acc_host = NULL;
static cl_context context = NULL;
static cl_command_queue command_queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;

static bool inited = false;
static bool built = false;
static int N_store = 0;
static int pos_size = 0;

#define xstr(s) str(s)
#define str(s) #s
#define xcomb(a,b) comb(a,b)
#define comb(a,b) a##b

#define EPSILON 1e-200;
#define VAR_NAME force_kernel
#define FILE_EXT ".cl"
#define LEN_EXT _len
#define VAR_LENGTH xcomb(VAR_NAME,LEN_EXT)
#define FILE_NAME xstr(VAR_NAME) FILE_EXT

#if USE_SHARED
static int numBlocks = 0;
static int sharedMemSize = 0;
#define BLOCK_X 32
#define THREADS_PER_BODY 8

#if THREADS_PER_BODY == 1
#define KERNEL_NAME "calculate_force_shared"
#else
#define KERNEL_NAME "calculate_force_shared_MT"
#endif
#else
#define KERNEL_NAME "calculate_force"
#endif

void opencl_build(void);
void opencl_clear_buffers(void);
void check_ret(char* text, cl_int ret);

//#define DUMP_DATA
#ifdef DUMP_DATA
void dump_data(int n);
#endif

void opencl_init(int N) 
{
    if (inited && N==N_store) 
        return;

    printf("  opencl_init N=%d, ", N);
    opencl_build();

    // Clean up anything used previously
    opencl_clear_buffers();

#if USE_SHARED
    sharedMemSize = BLOCK_X * THREADS_PER_BODY * 4 * sizeof(cl_double4);
    numBlocks = ((int)N + BLOCK_X - 1) / BLOCK_X;

    // Create a new data set - need to round up the size
    // of the pos block to multiple of block size and zero it (so mass is 0)
    pos_size = numBlocks * BLOCK_X;
    printf("Pos Size: %i, ", pos_size);
#else
    pos_size = N;
#endif

    // Allocate the data blocks
    cl_int ret = 0;
    pos_host = (double*)malloc(pos_size * 4 * sizeof(double));
    if (!pos_host) { printf("malloc failure\n"); exit(0); }

    acc_host = (double*)malloc(pos_size * 4 * sizeof(double));
    if (!acc_host) { printf("malloc failure\n"); exit(0); }

    // Set to zero, so that can be used to zero the OpenCL buffers
    memset(pos_host, 0, pos_size * 4 * sizeof(double));

    pos_dev = clCreateBuffer(context, CL_MEM_READ_ONLY,
                             pos_size * sizeof(cl_double4), NULL, &ret);
    check_ret("clCreateBuffer Pos", ret);
    acc_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                             pos_size * sizeof(cl_double4), NULL, &ret);
    check_ret("clCreateBuffer Acc", ret);

    // Clear the host memory
    ret = clEnqueueWriteBuffer(command_queue, pos_dev, CL_TRUE, 0,
                               pos_size * sizeof(cl_double4), 
                               pos_host, 0, NULL, NULL);
    check_ret("clEnqueueWriteBuffer", ret);

    inited = true;
    N_store = N;

#if USE_SHARED
    printf("OpenCL force SHARED opened B=%u T=%u.\n", BLOCK_X, THREADS_PER_BODY);
#else
    printf("OpenCL force opened.\n");
#endif
}

void opencl_build(void)
{
    if (!built)
    {
        built = true;

        // Get platform and device information
        cl_platform_id platform_id[MAX_PLATFORMS];
        cl_device_id device_id = NULL;
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;

        cl_int ret = clGetPlatformIDs(MAX_PLATFORMS, platform_id, &ret_num_platforms);
        check_ret("clGetPlatformIDs", ret);
        printf("Num Platforms: %u, ", ret_num_platforms);

        ret = -1;

        // Find the first platform with a GPU
        for (int i = 0; i < ret_num_platforms; ++i)
        {
            ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU, 1,
                                 &device_id, &ret_num_devices);

            if (ret == CL_SUCCESS)
            {
                break;
            }
        }

        if (ret != CL_SUCCESS)
        {
            printf("No suitable device found. Error %i\n", ret);
            exit(0);
        }

        // Have a valid device_id, create an OpenCL context
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        check_ret("clCreateContext", ret);

        // Create a command queue
        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
        check_ret("clCreateCommandQueue", ret);

        // Load the kernel source code into the array source_str
        char* source_str;
        size_t source_size;

#ifdef LOAD_FROM_STRING
        source_str = VAR_NAME;
        source_size = VAR_LENGTH;
#else
        FILE* fp = fopen(FILE_NAME, "r");
        if (!fp)
        {
            fprintf(stderr, "Failed to load kernel. %s\n", FILE_NAME);
            exit(0);
        }

        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);
#endif
        // Create a program from the kernel source
        program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
                                            (const size_t*)&source_size, &ret);
        check_ret("clCreateProgramWithSource", ret);

        // Build the program
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

        if (ret != CL_SUCCESS)
        {
            char buffer[2048];
            size_t length;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
            printf("Build log\n%s", buffer);
            check_ret("clBuildProgram", ret);
        }

        // Create the OpenCL kernel
        kernel = clCreateKernel(program, KERNEL_NAME, &ret);
        check_ret("clCreateKernel", ret);

#ifndef LOAD_FROM_STRING
        free(source_str);
#endif
    }
}

void opencl_finalize() 
{
    if (built)
    {
        // Clear down the rest of OpenCL
        cl_int ret = clFinish(command_queue);
        check_ret("clFinish", ret);

        ret = clReleaseKernel(kernel);
        check_ret("clReleaseKernel", ret);

        ret = clReleaseProgram(program);
        check_ret("clReleaseProgram", ret);

        opencl_clear_buffers();

        ret = clReleaseCommandQueue(command_queue);
        check_ret("clReleaseCommandQueue", ret);

        ret = clReleaseContext(context);
        check_ret("clReleaseContext", ret);

        built = false;
    }
    else
    {
        opencl_clear_buffers();
    }
}

void opencl_clear_buffers()
{
    cl_int ret;

    if (pos_host)
        printf("Opencl clear buffers...\n");
    if (pos_dev != NULL)
    {
        ret = clReleaseMemObject(pos_dev);
        if (ret != CL_SUCCESS) exit(0);
    }
    
    if (acc_dev != NULL)
    {
        ret = clReleaseMemObject(acc_dev);
        if (ret != CL_SUCCESS) exit(0);
    }
    
    free(pos_host);
    free(acc_host);

    pos_dev = NULL;
    acc_dev = NULL;
    pos_host = NULL;
    acc_host = NULL;
    inited = false;
    N_store = 0;
}

void check_ret(char* text, cl_int ret)
{
    if (ret != CL_SUCCESS)
    {
        printf("%s: Error =%i\n", text, ret);
        exit(-1);
    }
}

size_t ode_n_body_second_order_opencl(const real vec[], size_t N, real G, const real masses[], const real radii[], real acc[]) {
    if (masses == NULL) {printf("masses=NULL, exiting...\n"); exit(0);}

    cl_int ret;
    double eps = EPSILON;
    int n = (int)N;

    opencl_init(n);

    for (int i = 0; i < n; i++) {
        pos_host[4 * i] = vec[3 * i];
        pos_host[4 * i + 1] = vec[3 * i + 1];
        pos_host[4 * i + 2] = vec[3 * i + 2];
        pos_host[4 * i + 3] = masses[i] * G;
    }

    size_t global_work_size[2];
    size_t local_work_size[2];

    ret = clEnqueueWriteBuffer(command_queue, pos_dev, CL_TRUE, 0,
                               n * 4 * sizeof(double), pos_host, 0, NULL, NULL);
    check_ret("clEnqueueWriteBuffer", ret);

    // Pass the parameters
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&pos_dev);
    check_ret("clSetKernelArg 0", ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&acc_dev);
    check_ret("clSetKernelArg 1", ret);
#if USE_SHARED
    ret = clSetKernelArg(kernel, 2, sharedMemSize, NULL);
    check_ret("clSetKernelArg 4", ret);
#else
    ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&n);
    check_ret("clSetKernelArg 2", ret);
#endif
    ret = clSetKernelArg(kernel, 3, sizeof(double), (void*)&eps);
    check_ret("clSetKernelArg 3", ret);

#if USE_SHARED
    // set work-item dimensions
    local_work_size[0] = BLOCK_X;
    local_work_size[1] = THREADS_PER_BODY;
    global_work_size[0] = pos_size;
    global_work_size[1] = THREADS_PER_BODY;
#else
    local_work_size[0] = 1;
    local_work_size[1] = 1;
    global_work_size[0] = n;
    global_work_size[1] = 1;
#endif

    // execute the kernel using shared memory:
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                 global_work_size, local_work_size, 0, NULL, NULL);
    check_ret("clEnqueueNDRangeKernel", ret);

    // Read the accelartion memory buffer on the device 
    // This implicitly flushes the command queue
    ret = clEnqueueReadBuffer(command_queue, acc_dev, CL_TRUE, 0,
                              n * 4 * sizeof(double), acc_host, 0, NULL, NULL);
    check_ret("clEnqueueReadBuffer", ret);

    // Reformat back into the acc buffer
    for (int i = 0; i < n; i++) {
        acc[3 * i] = acc_host[4 * i];
        acc[3 * i + 1] = acc_host[4 * i + 1];
        acc[3 * i + 2] = acc_host[4 * i + 2];
    }

#ifdef DUMP_DATA
    dump_data(n);
#endif
    return 0;
}

#ifdef DUMP_DATA
void dump_data(int n)
{
    static int count = 0;

    if (++count == 1)
    {
        for (int i=0;i<n;i++)
        {
            printf("i: %i, accx %e, accy %e accz %e\n", i, acc_host[4 * i], acc_host[4 * i + 1], acc_host[4 * i + 2]);
        }
        exit(0);
    }
}
#endif

#endif
