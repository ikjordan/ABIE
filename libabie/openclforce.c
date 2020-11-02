#ifdef OPENCL

//#define USE_SHARED
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl_ext.h>
#include "common.h"

#define BLOCK_SIZE 32
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
static int blockSize = BLOCK_SIZE;
static int numBlocks = 0;

#define EPSILON 1e-200;

#ifdef USE_SHARED
int sharedMemSize = 0;
#endif

void opencl_build(void);
void opencl_clear_buffers(void);
void check_ret(char* text, cl_int ret);

void opencl_init(int N) 
{
    if (inited && N==N_store) 
        return;

    printf("  opencl_init N=%d  ", N);
    opencl_build();

    // Clean up anything used previously
    opencl_clear_buffers();

    numBlocks = ((int)N + blockSize - 1) / blockSize;

#ifdef USE_SHARED
    // Create a new data set - need to round up the size
    // of the pos block to multiple of block size and zero it (so mass is 0)
    int pos_size = numBlocks * blockSize;
    sharedMemSize = blockSize * sizeof(double4); // 4 doubles for pos
#else
    int pos_size = N;
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
                             pos_size * 4 * sizeof(double), NULL, &ret);
    check_ret("clCreateBuffer Pos", ret);
    acc_dev = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                             pos_size * 4 * sizeof(double), NULL, &ret);
    check_ret("clCreateBuffer Acc", ret);

    // Clear the host memory
    ret = clEnqueueWriteBuffer(command_queue, pos_dev, CL_TRUE, 0,
                               pos_size * 4 * sizeof(double), 
                               pos_host, 0, NULL, NULL);
    check_ret("clEnqueueWriteBuffer", ret);

    inited = true;
    N_store = N;

#ifdef USE_SHARED
    err = setSofteningSquared(EPSILON);
    if (err > 0) { printf("setSofteningSquared err = %d\n", err); exit(0); }
    printf("...GPU force SHARED opened. ");
#else
    printf("...GPU force opened. ");
#endif
    printf("blocksize = %d\n", blockSize);
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
        printf("Num Platforms: %u\n\n", ret_num_platforms);

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
        FILE* fp;
        char* source_str;
        size_t source_size;

        fp = fopen("force_kernel.cl", "r");
        if (!fp)
        {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(0);
        }

        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        // Create a program from the kernel source
        program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
                                            (const size_t*)&source_size, &ret);
        check_ret("clCreateProgramWithSource", ret);

        // Build the program
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        check_ret("clBuildProgram", ret);

        // Create the OpenCL kernel
        kernel = clCreateKernel(program, "calculate_force", &ret);
        check_ret("clCreateKernel", ret);

        free(source_str);
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

    ret = clEnqueueWriteBuffer(command_queue, pos_dev, CL_TRUE, 0,
                               n * 4 * sizeof(double), pos_host, 0, NULL, NULL);
    check_ret("clEnqueueWriteBuffer", ret);

#ifdef USE_SHARED
    gpuforce_shared<<<numBlocks, blockSize, sharedMemSize >>>(pos_dev, (int)N, acc_dev, numBlocks);
#else   
    // Pass the parameters
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&pos_dev);
    check_ret("clSetKernelArg 0", ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&acc_dev);
    check_ret("clSetKernelArg 1", ret);
    ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&n);
    check_ret("clSetKernelArg 2", ret);
    ret = clSetKernelArg(kernel, 3, sizeof(double), (void*)&eps);
    check_ret("clSetKernelArg 3", ret);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = n;    // Process all n
    size_t local_item_size = 1;     // each body separately
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                 &global_item_size, &local_item_size, 0, NULL, NULL);
    check_ret("clEnqueueNDRangeKernel", ret);

    // Read the accelartion memory buffer on the device 
    // This implicitly flushes the comman queue
    ret = clEnqueueReadBuffer(command_queue, acc_dev, CL_TRUE, 0,
                              n * 4 * sizeof(double), acc_host, 0, NULL, NULL);
    check_ret("clEnqueueReadBuffer", ret);

    // Reformat back into the acc buffer
    for (int i = 0; i < n; i++) {
        acc[3 * i] = acc_host[4 * i];
        acc[3 * i + 1] = acc_host[4 * i + 1];
        acc[3 * i + 2] = acc_host[4 * i + 2];
    }

#if 0
    for (int i=0;i<min(n,5);++i)
    {
        printf("%i: acc.x: %f, acc.y: %f, acc.z: %f\n", i, acc_host[4 * i], acc_host[4 * i + 1], acc_host[4 * i + 2]);
    }
    for (int i = max(n-5,0); i < n; ++i)
    {
        printf("%i: acc.x: %f, acc.y: %f, acc.z: %f\n", i, acc_host[4 * i], acc_host[4 * i + 1], acc_host[4 * i + 2]);
    }
    exit(0);
#endif
    return 0;
#endif
}

#endif
