#ifdef GPU

extern "C" {
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "common.h"
}

#ifdef USE_SHARED
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<class T>
struct SharedMemory
{
    __device__ inline operator T* ()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T* () const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};
#endif

#define BLOCK_SIZE 32

double4 *pos_dev = NULL;
double3 *acc_dev = NULL;
double* pos_host = NULL;
int inited = 0;
int N_store = 0;
int blockSize = BLOCK_SIZE;
int numBlocks = 0;

#ifdef USE_SHARED
int sharedMemSize = 0;
#define EPSILON 1e-200

__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(double softeningSq)
{
    return cudaMemcpyToSymbol(softeningSquared_fp64,
                              &softeningSq,
                              sizeof(double), 0,
                              cudaMemcpyHostToDevice);
};

__device__ double getSofteningSquared()
{
    return softeningSquared_fp64;
}

__device__ double3 bodyBodyInteraction(double3 ai, double4 bi, double4 bj)
{
    double3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z;

    // Softening is so small that it only impacts the case when bj = bi
    // In that case the evaluates acceleration is 0, as bj - bj = 0
    distSqr += getSofteningSquared();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    double invDist = rsqrt(distSqr);
    double invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    double s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ double3 computeBodyAccel(double4 bodyPos,
                                    double4* positions,
                                    int numTiles, 
                                    cg::thread_block cta)
{
    double4* sharedPos = SharedMemory<double4>();

    double3 acc = { 0.0f, 0.0f, 0.0f };

    for (int tile = 0; tile < numTiles; tile++)
    {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        cg::sync(cta);

#pragma unroll BLOCK_SIZE
        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter]);
        }
        cg::sync(cta);
    }
    return acc;
}

__global__ void gpuforce_shared(double4* __restrict__ p, int n, double3* __restrict__ acc, int numTiles) {
    cg::thread_block cta = cg::this_thread_block();
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Need to execute, even if i>=n as shared memory needs to be fully initialised
    double3 accel = computeBodyAccel(p[i], p, numTiles, cta);
    if (i < n)
    {
        acc[i].x = accel.x;
        acc[i].y = accel.y;
        acc[i].z = accel.z;
    }
}

#else
__global__ void gpuforce(double4* __restrict__ p, int n, double3* __restrict__ acc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        double3 F = { 0.0f, 0.0f, 0.0f };

 //#pragma unroll
        for (int j = 0; j < n; j++) {
            double m = p[j].w;
            if (i == j || m == 0) continue;
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double dz = p[i].z - p[j].z;
            double distSqr = dx * dx + dy * dy + dz * dz;
            double invDist = rsqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            F.x -= (m * dx * invDist3);
            F.y -= (m * dy * invDist3);
            F.z -= (m * dz * invDist3);
        }
        acc[i].x = F.x;
        acc[i].y = F.y;
        acc[i].z = F.z;
    }
}
#endif
extern "C" {
    void gpu_init(int N) {
        if (inited && N==N_store) return;
        // Clean up anything used previously
        printf("  gpu_init N=%d  ", N);
        gpu_finalize();

        numBlocks = ((int)N + blockSize - 1) / blockSize;

#ifdef USE_SHARED
        // Create a new data set - need to round up the size
        // of the pos block to multiple of block size and zero it (so mass is 0)
        int pos_size = numBlocks * blockSize;
        sharedMemSize = blockSize * sizeof(double4); // 4 doubles for pos
#else
        int pos_size = N;
#endif

        int err = 0;
        err = cudaMalloc(&pos_dev, pos_size * sizeof(double4));
        if (err > 0) {printf("cudaMalloc err = %d\n", err); exit(0); }
        err = cudaMemset(pos_dev, 0, pos_size * sizeof(double4));
        if (err > 0) { printf("cudaMemset err = %d\n", err); exit(0); }
        err = cudaMalloc(&acc_dev, N * sizeof(double3));
        if (err > 0) {printf("cudaMalloc err = %d\n", err); exit(0); }
        pos_host = (double*)malloc(N * 4 * sizeof(double));
        if (err > 0) { printf("cudaMalloc err = %d\n", err); exit(0); }

        inited = 1;
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

    void gpu_finalize() {
        if (pos_host)
            printf("Closing GPU force...\n");
        if (pos_dev != NULL) cudaFree(pos_dev);
        if (acc_dev != NULL) cudaFree(acc_dev);
        free(pos_host);
        pos_dev = NULL;
        acc_dev = NULL;
        pos_host = NULL;
        inited = 0;
        N_store = 0;
    }

    size_t ode_n_body_second_order_gpu(const real vec[], size_t N, real G, const real masses[], const real radii[], real acc[]) {
        if (masses == NULL) {printf("masses=NULL, exiting...\n"); exit(0);}

        cudaError_t err;
        gpu_init((int)N);

        for (size_t i = 0; i < N; i++) {
            pos_host[4 * i] = vec[3 * i];
            pos_host[4 * i + 1] = vec[3 * i + 1];
            pos_host[4 * i + 2] = vec[3 * i + 2];
            pos_host[4 * i + 3] = masses[i] * G;
        }

        err = cudaMemcpy(pos_dev, pos_host, N*sizeof(double4), cudaMemcpyHostToDevice);
        if (err > 0) {printf("cudaMemcpy err = %d, host_to_dev\n", err); exit(0); }

#ifdef USE_SHARED
        gpuforce_shared<<<numBlocks, blockSize, sharedMemSize >>>(pos_dev, (int)N, acc_dev, numBlocks);
#else   

        gpuforce<<<numBlocks, blockSize >>>(pos_dev, (int)N, acc_dev);
#endif
        err = cudaGetLastError();
        if (err != cudaSuccess) {printf("Error: %d %s\n", err, cudaGetErrorString(err)); exit(0);}


        // err = cudaMemcpy(acc_host, acc_dev, bytes, cudaMemcpyDeviceToHost);
        err = cudaMemcpy(acc, acc_dev, N*sizeof(double3), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {printf("cudaMemcpy err = %d, %s\n", err, cudaGetErrorString(err)); exit(0); }

        return 0;
    }
} // end extern C

#endif
