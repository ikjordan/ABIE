// Note the shared kernels assume that the number of bodies is a multiple 
// of the local block size (typically 32)
// If this is not the case, then dummy particles of 0 mass have to be added 
// before calling the shared kernels

double3 bodyBodyInteraction(double3 ai, 
                            double4 bi, 
                            double4 bj, 
                            double softening_sqr)
{
    double3 r;

    // r_ij  [3 FLOPS]
    r.xyz = bi.xyz - bj.xyz;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    double distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += softening_sqr;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    double invDist = rsqrt(distSqr);
    double invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    double s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.xyz -= r.xyz * s;

    return ai;
}

#define UNROLL_FACTOR 8

__kernel void calculate_force(__global double4* pos,
                              __global double3* accel, 
                              int num_bodies,
                              double softening_sqr)
{
    double3 acc = { 0.0f, 0.0f, 0.0f };
    unsigned int global_id = get_global_id(0);
    double4 my_pos = pos[global_id];

    unsigned int i = 0;

    for (; (i + UNROLL_FACTOR) < num_bodies; ) 
    {
#pragma unroll UNROLL_FACTOR
        for(int j = 0; j < UNROLL_FACTOR; j++, i++) 
        {
            acc = bodyBodyInteraction(acc, my_pos, pos[i], softening_sqr);
        }
    }

    // In case the number of bodies is not a multiple of UNROLL_FACTOR
    for (; i < num_bodies; i++) 
    {
        acc = bodyBodyInteraction(acc, my_pos, pos[i], softening_sqr);
    }
    accel[global_id] = acc;
}

double3 computeBodyAccel(double4 body_pos,
                         __global double4* positions,
                         double softening_sqr,
                         __local double4* shared_pos)
{
    double3 acc = { 0.0f, 0.0f, 0.0f };
    unsigned int block_dim_x = get_local_size(0);
    unsigned int thread_id_xx = get_local_id(0);
    unsigned int num_tiles = get_num_groups(0);

    for (int tile = 0; tile < num_tiles; tile++)
    {
        shared_pos[thread_id_xx] = positions[tile * block_dim_x + thread_id_xx];

        // Populate all entries in shared memory before proceeding
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int counter = 0; counter < block_dim_x; counter++)
        {
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[counter], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter], softening_sqr);
        }
        // All threads must have finished accesses to shared memory before proceeding
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return acc;
}

__kernel void calculate_force_shared(__global double4* pos,
                                     __global double3* accel,
                                     __local double4* shared_pos,
                                     double softening_sqr)
{
    unsigned int block_dim_x = get_local_size(0);
    unsigned int block_id_xx = get_group_id(0);
    unsigned int thread_id_xx = get_local_id(0);

    int i = block_dim_x * block_id_xx + thread_id_xx;

    accel[i] = computeBodyAccel(pos[i], pos, softening_sqr, shared_pos);
}

double3 computeBodyAccel_MT(double4 body_pos,
                            __global double4* positions,
                            double softening_sqr,
                            __local double4* shared_pos)
{
    // Divide the calculation of a tile across multiple threads
    double3 acc = { 0.0f, 0.0f, 0.0f };
    unsigned int block_dim_x = get_local_size(0);
    unsigned int block_dim_y = get_local_size(1);
    unsigned int thread_id_xx = get_local_id(0);
    unsigned int thread_id_xy = get_local_id(1);
    unsigned int num_tiles = get_num_groups(0);
    unsigned int offset = mul24(block_dim_x, thread_id_xy); 

    // First thread takes the first block_dim_x / block_dim_y positions
    for (unsigned int tile = 0; tile < num_tiles; tile += block_dim_y)
    {
        shared_pos[thread_id_xx + offset] = 
            positions[mul24(tile, block_dim_x) + thread_id_xx + offset];

        // Populate all entries in shared memory before proceeding
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int counter = 0; counter < block_dim_x; counter++)
        {
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[++counter + offset], softening_sqr);
        }
        // All threads must have finished accesses to shared memory before proceeding
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Now write back the result to the shared memory
    shared_pos[thread_id_xx + offset].x = acc.x;
    shared_pos[thread_id_xx + offset].y = acc.y;
    shared_pos[thread_id_xx + offset].z = acc.z;

    barrier(CLK_LOCAL_MEM_FENCE);

    // All data now in shared memory, so can be reduced
    if (thread_id_xy == 0)
    {
        for (unsigned int i = 1; i < block_dim_y; i++)
        {
            acc.x += shared_pos[thread_id_xx + mul24(block_dim_x, i)].x;
            acc.y += shared_pos[thread_id_xx + mul24(block_dim_x, i)].y;
            acc.z += shared_pos[thread_id_xx + mul24(block_dim_x, i)].z;
        }
    }
    return acc;
}

__kernel void calculate_force_shared_MT(__global double4* pos,
                                        __global double3* accel,
                                        __local double4* shared_pos,
                                        double softening_sqr)
{
    unsigned int block_dim_x = get_local_size(0);
    unsigned int block_id_xx = get_group_id(0);
    unsigned int thread_id_xx = get_local_id(0);
    unsigned int thread_id_xy = get_local_id(1);

    int i = block_dim_x * block_id_xx + thread_id_xx;

    // Need to execute, even if i>=n as shared memory needs to be fully initialised
    double3 acc = computeBodyAccel_MT(pos[i], pos, softening_sqr, shared_pos);

    // Write back the aggregated calculation
    if (thread_id_xy == 0)
    {
        accel[i] = acc;
    }
}
