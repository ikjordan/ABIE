/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each work-item invocation of this kernel, calculates the position for 
 * one particle
 *
 */

#define UNROLL_FACTOR  8
__kernel 
void calculate_force(__global double4* pos, __global double3* new_acc,
                     unsigned int numBodies, double epsSqr) 
{

    unsigned int gid = get_global_id(0);
    double4 myPos = pos[gid];
    double3 acc = (double3)0.0;


    unsigned int i = 0;
    for (; (i+UNROLL_FACTOR) < numBodies; ) {
#pragma unroll UNROLL_FACTOR
        for(int j = 0; j < UNROLL_FACTOR; j++,i++) {
            double4 p = pos[i];
            double4 r;
            r.xyz = p.xyz - myPos.xyz;
            double distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;

            double invDist = rsqrt(distSqr + epsSqr);
            double invDistCube = invDist * invDist * invDist;
            double s = p.w * invDistCube;

            // accumulate effect of all particles
            acc.xyz += s * r.xyz;
        }
    }
    for (; i < numBodies; i++) {
        double4 p = pos[i];

        double4 r;
        r.xyz = p.xyz - myPos.xyz;
        double distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;

        double invDist = rsqrt(distSqr + epsSqr);
        double invDistCube = invDist * invDist * invDist;
        double s = p.w * invDistCube;

        // accumulate effect of all particles
        acc.xyz += s * r.xyz;
    }

    // write to global memory
    new_acc[gid] = acc;
}
