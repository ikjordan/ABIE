// Note these routines assume the the number of 
// bodies is a multiple of the local block size
// (typically 32)

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

double3 computeBodyAccel(double4 bodyPos,
                         __global double4* positions,
                         double softening_sqr,
                         __local double4* sharedPos)
{
    double3 acc = { 0.0f, 0.0f, 0.0f };
    unsigned int blockDimx = get_local_size(0);
    unsigned int threadIdxx = get_local_id(0);
    unsigned int num_tiles = get_num_groups(0);

    for (int tile = 0; tile < num_tiles; tile++)
    {
        sharedPos[threadIdxx] = positions[tile * blockDimx + threadIdxx];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int counter = 0; counter < blockDimx; counter++)
        {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter], softening_sqr);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return acc;
}

double3 computeBodyAccel2(double4 bodyPos,
                          __global double4* positions,
                          double softening_sqr,
                          __local double4* sharedPos)
{
    // Divide the calculation of a tile across multiple threads
    double3 acc = { 0.0f, 0.0f, 0.0f };
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockDimy = get_local_size(1);
    unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);
    unsigned int num_tiles = get_num_groups(0);
    unsigned int offset = mul24(blockDimx, threadIdxy); 

    // First thread takes the first blockDimx / blockDimy positions
    for (unsigned int tile = 0; tile < num_tiles; tile += blockDimy)
    {
        sharedPos[threadIdxx + offset] = 
            positions[mul24(tile, blockDimx) + threadIdxx + offset];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int counter = 0; counter < blockDimx; counter++)
        {
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter + offset], softening_sqr);
            acc = bodyBodyInteraction(acc, bodyPos, sharedPos[++counter + offset], softening_sqr);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Now write back the result to the shared memory
    sharedPos[threadIdxx + offset].x = acc.x;
    sharedPos[threadIdxx + offset].y = acc.y;
    sharedPos[threadIdxx + offset].z = acc.z;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Sum the results
    if (threadIdxy == 0)
    {
        for (unsigned int i = 1; i < blockDimy; i++)
        {
            acc.x += sharedPos[threadIdxx + mul24(blockDimx, i)].x;
            acc.y += sharedPos[threadIdxx + mul24(blockDimx, i)].y;
            acc.z += sharedPos[threadIdxx + mul24(blockDimx, i)].z;
        }
    }
    return acc;
}

__kernel void calculate_force_shared(__global double4* pos,
                                     __global double3* accel, 
                                     int numBodies,
                                     double softening_sqr,
                                     __local double4* sharedPos)
{
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int threadIdxx = get_local_id(0);

    int i = blockDimx * blockIdxx + threadIdxx;

    // Need to execute, even if i>=n as shared memory needs to be fully initialised
    accel[i] = computeBodyAccel(pos[i], pos, softening_sqr, sharedPos);
}

__kernel void calculate_force_shared_MT(__global double4* pos,
                                        __global double3* accel,
                                        int numBodies,
                                        double softening_sqr,
                                        __local double4* sharedPos)
{
    unsigned int blockDimx = get_local_size(0);
    unsigned int blockIdxx = get_group_id(0);
    unsigned int threadIdxx = get_local_id(0);
    unsigned int threadIdxy = get_local_id(1);

    int i = blockDimx * blockIdxx + threadIdxx;

    // Need to execute, even if i>=n as shared memory needs to be fully initialised
    double3 acc = computeBodyAccel2(pos[i], pos, softening_sqr, sharedPos);

    // Write back the aggregated calculation
    if (threadIdxy == 0)
    {
        accel[i] = acc;
    }
}
