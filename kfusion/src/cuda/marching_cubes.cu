#include "device.hpp"
#include "texture_binder.hpp"

namespace kfusion
{
    namespace device
    {
        //texture<int, 1, cudaReadModeElementType> edgeTex;
        texture<int, 1, cudaReadModeElementType> triTex;
        texture<int, 1, cudaReadModeElementType> numVertsTex;
    }
}

void
kfusion::device::bindTextures (const int */*edgeBuf*/, const int *triBuf, const int *numVertsBuf)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    //cudaSafeCall(cudaBindTexture(0, edgeTex, edgeBuf, desc) );
    cudaSafeCall (cudaBindTexture (0, triTex, triBuf, desc) );
    cudaSafeCall (cudaBindTexture (0, numVertsTex, numVertsBuf, desc) );
}

void
kfusion::device::unbindTextures ()
{
    //cudaSafeCall( cudaUnbindTexture(edgeTex) );
    cudaSafeCall ( cudaUnbindTexture (numVertsTex) );
    cudaSafeCall ( cudaUnbindTexture (triTex) );
}

namespace kfusion
{
    namespace device
    {
        __device__ int global_count = 0;
        __device__ int output_count;
        __device__ unsigned int blocks_done = 0;

        struct CubeIndexEstimator
        {
            TsdfVolume volume;

            CubeIndexEstimator(const TsdfVolume& _volume) : volume(_volume) {}

            static __kf_device__ float isoValue () {return 0.f;}

            __kf_device__
            void readTsdf (int x, int y, int z, float &tsdf, int &weight) const
            {
                tsdf = unpack_tsdf(*volume(x, y, z), weight);
            }

            __kf_device__
            int computeCubeIndex (int x, int y, int z, float f[8]) const
            {
                int weight;
                readTsdf(x, y, z, f[0], weight);
                if (weight == 0) return 0;
                readTsdf(x + 1, y, z, f[1], weight);
                if (weight == 0) return 0;
                readTsdf(x + 1, y + 1, z, f[2], weight);
                if (weight == 0) return 0;
                readTsdf(x, y + 1, z, f[3], weight);
                if (weight == 0) return 0;
                readTsdf(x, y, z + 1, f[4], weight);
                if (weight == 0) return 0;
                readTsdf(x + 1, y, z + 1, f[5], weight);
                if (weight == 0) return 0;
                readTsdf(x + 1, y + 1, z + 1, f[6], weight);
                if (weight == 0) return 0;
                readTsdf(x, y + 1, z + 1, f[7], weight);
                if (weight == 0) return 0;

                // calculate flag indicating if each vertex is inside or outside isosurface
                int cubeindex;
                cubeindex = int(f[0] < isoValue());
                cubeindex += int(f[1] < isoValue()) * 2;
                cubeindex += int(f[2] < isoValue()) * 4;
                cubeindex += int(f[3] < isoValue()) * 8;
                cubeindex += int(f[4] < isoValue()) * 16;
                cubeindex += int(f[5] < isoValue()) * 32;
                cubeindex += int(f[6] < isoValue()) * 64;
                cubeindex += int(f[7] < isoValue()) * 128;

                return cubeindex;
            }
        };

        struct OccupiedVoxels : public CubeIndexEstimator
        {
            enum
            {
                CTA_SIZE_X = 32,
                CTA_SIZE_Y = 8,
                CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

                WARPS_COUNT = CTA_SIZE / Warp::WARP_SIZE
            };

            mutable int* voxels_indices;
            mutable int* vertices_number;
            int max_size;

            OccupiedVoxels(const TsdfVolume& _volume) : CubeIndexEstimator(_volume) {}

            __kf_device__
            void operator () () const
            {

            }
        };
        __global__ void getOccupiedVoxelsKernel (const OccupiedVoxels ov) {ov();}
    } // namespace device
} // namespace kfusion

int
kfusion::device::getOccupiedVoxels (const TsdfVolume& volume, DeviceArray2D<int>& occupied_voxels)
{
    OccupiedVoxels ov(volume);

    ov.voxels_indices = occupied_voxels.ptr (0);
    ov.vertices_number = occupied_voxels.ptr (1);
    ov.max_size = occupied_voxels.cols ();

    dim3 block (OccupiedVoxels::CTA_SIZE_X, OccupiedVoxels::CTA_SIZE_Y);
    dim3 grid (divUp (volume.dims.x, block.x), divUp (volume.dims.y, block.y));

    //cudaFuncSetCacheConfig(getOccupiedVoxelsKernel, cudaFuncCachePreferL1);
    //printFuncAttrib(getOccupiedVoxelsKernel);

    getOccupiedVoxelsKernel<<<grid, block>>>(ov);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());

    int size;
    cudaSafeCall ( cudaMemcpyFromSymbol (&size, output_count, sizeof(size)) );
    return size;
}