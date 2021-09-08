#include "Tensor.cginc"
#define UNITY_SHADER_NO_UPGRADE 1

TENSOR_DECL(X)
TENSOR_DECL(K)
TENSOR_DECL(B)
TENSOR_DECL(WBK)
TENSOR_DECL_RW(O)

uint4 _Pad;
uint4 _Stride;

#define DEBUG_CHECK_BOUNDS 0

// Conv2DBlock64x64_4x4 + index optimizations
//        T
//      -1|0             -1|0
// 16: 142|142ms        144|155ms

float ffma(float a, float b, float c) { return dot(float2(a,c), float2(b,1)); }

#if CHANNELS_FIRST
    #define FUNC_NAME_CALL(KERNEL, SUFFIX, SIZE) KERNEL##SUFFIX##SIZE##x##SIZE##_NCHW
    #define CACHE_NAME_CALL(KERNEL, SUFFIX, SIZE, TENSOR) KERNEL##SUFFIX##SIZE##x##SIZE##_Cache_##TENSOR##_NCHW
#else
    #define FUNC_NAME_CALL(KERNEL, SUFFIX, SIZE) KERNEL##SUFFIX##SIZE##x##SIZE##_NHWC
    #define CACHE_NAME_CALL(KERNEL, SUFFIX, SIZE, TENSOR) KERNEL##SUFFIX##SIZE##x##SIZE##_Cache_##TENSOR##_NHWC
#endif
#define FUNC_NAME(KERNEL, SUFFIX, SIZE) FUNC_NAME_CALL(KERNEL, SUFFIX, SIZE)
#define CACHE_NAME(KERNEL, SUFFIX, SIZE, TENSOR) CACHE_NAME_CALL(KERNEL, SUFFIX, SIZE, TENSOR)

#define KERNEL_NAME Conv2D

#if BLOCK_SIZE == 8
#if KERNEL_PER_TG == 64

#if CHANNELS_FIRST
    //NCHW
    #define CACHE_DEPTH 8                      // Profiled as the fastest to avoid 'tail' of inner loops with occupancy 1 at end of dispatch.
    #define CACHE_WIDTH_W_PAD 1
    #define NUM_DDR_LOAD_PER_LOOP CACHE_DEPTH  // Not needed for NCHW
    #define SHUFFLE_FOR_COALESCED_LOAD 0       // Not needed for NCHW
    #define SHUFFLE_FOR_COALESCED_STORE 1
#else
    //NHWC
    #define CACHE_DEPTH 16                     // Only supported value
    #define CACHE_WIDTH_W_PAD 0                // Only supported value
    #define NUM_DDR_LOAD_PER_LOOP 8            // <=8 required to lower register pressure for NHWC for occupancy of 2.
    #define SHUFFLE_FOR_COALESCED_LOAD 1
    #define SHUFFLE_FOR_COALESCED_STORE 1
#endif
#define CACHE_WIDTH_X 64                       // Only supported value
#define CACHE_WIDTH_W (64+CACHE_WIDTH_W_PAD)   // Only supported value

#if SHUFFLE_FOR_COALESCED_STORE
    //A TG output [64pixels,64channels] = 4096 values, We will write two time 2048 values to DDR (8k LDS).
    groupshared float CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, LDS)[2048];
#else
    groupshared float CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, LDS)[CACHE_DEPTH*(CACHE_WIDTH_X+CACHE_WIDTH_W)];
#endif

[numthreads(8,8,1)]
void FUNC_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE)(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID, uint threadIndex : SV_GroupIndex)
{
    //This kernel assume the following:
    //Input:
    //  C % CACHE_DEPTH==0 <-- only if STRICT_CHANNELS==1
    //Ouput:
    //  W%4==0 <-- only if CHANNELS_FIRST==1
    //Kernel:
    //  K%64==0 <-- only if LAX_KERNEL=0 else K%16==0 is required
    //DISPATCH ARGS(K.kernelCount, O.width * O.height, O.batch);
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);
    #define LDS_ CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, LDS)
    #define X_OFFSET 0
    #define W_OFFSET CACHE_DEPTH*CACHE_WIDTH_X

    //Per thread group (scalar registers)
    uint tg_NumChannels = X.channels;
    uint tg_WidthX  = X.width;
    uint tg_HeightX = X.height;
    uint tg_WidthO  = O.width;
    uint tg_HeightO = O.height;
    uint tg_NumKernels = K.channels;
    uint tg_NumInputPixels = tg_WidthX*tg_HeightX;
    uint tg_NumOuputPixels = tg_WidthO*tg_HeightO;
    uint tg_KernelSpatialStride = tg_NumKernels*tg_NumChannels;
    uint tg_KernelBaseId = groupID.x * CACHE_WIDTH_X;
    uint tg_OutputPixelBaseId = groupID.y * CACHE_WIDTH_X;
    uint tg_BatchReadOffset = groupID.z * tg_NumChannels * tg_HeightX * tg_WidthX;
    uint tg_BatchWriteOffset = groupID.z * tg_NumKernels * tg_HeightO * tg_WidthO;
    uint tg_kernelSpatialOffset = 0;

    //8x8 block, 8 kernels by 8 pixels
    //**********************************
    //* Kernel Ids  *  0  1  2  3  ...
    //**********************************
    //              *  ThreadIds
    // Pixel Ids  0 *  0  1  2  3 ...
    //            1 *  8  9 10 11 ...
    //            2 * 16 17 18 19 ...
    //            3 * 32 33 34 35 ...
    //            ... ...
    float dstA[BLOCK_SIZE*BLOCK_SIZE];

    //Load Bias [K] int dstA [Kernels, Pixels]
    uint tg_kId;
    uint tg_pId;
    uint maxBiasIndex = O.channels - 1;
    [unroll] for (tg_pId = 0; tg_pId < BLOCK_SIZE; ++tg_pId)
        [unroll] for (tg_kId = 0; tg_kId < BLOCK_SIZE; ++tg_kId)
            dstA[tg_pId*BLOCK_SIZE+tg_kId] = B.FastGet(min(maxBiasIndex,tg_KernelBaseId + groupThreadID.x * BLOCK_SIZE + tg_kId));

    for (uint tg_Dy = 0; tg_Dy < K.GetKernelHeight(); tg_Dy++)
    {
        for (uint tg_Dx = 0; tg_Dx < K.GetKernelWidth(); tg_Dx++)
        {
            for (uint tg_ChannelOffset = 0; tg_ChannelOffset < tg_NumChannels; tg_ChannelOffset += CACHE_DEPTH)
            {
                uint tg_CacheLoadDynIdx = 0;
                //Load from DDR to LDS: (64 weight + 64 pixel) * CACHE_DEPTH => 512Bytes * CACHE_DEPTH.
                //Storing in registers to avoid sync inside the loop.
                #if NUM_DDR_LOAD_PER_LOOP != CACHE_DEPTH
                for (; tg_CacheLoadDynIdx < CACHE_DEPTH/NUM_DDR_LOAD_PER_LOOP; ++tg_CacheLoadDynIdx)
                #endif
                {
                    //Explicit register declaration as [unroll] won't unroll properly otherwise and introduce sync points.
                    float tempW[NUM_DDR_LOAD_PER_LOOP];
                    float tempX[NUM_DDR_LOAD_PER_LOOP];
                    uint tg_regCacheLoadIdx;
                    [unroll] for (tg_regCacheLoadIdx = 0; tg_regCacheLoadIdx < NUM_DDR_LOAD_PER_LOOP; ++tg_regCacheLoadIdx)
                    {
                        uint tg_CacheLoadIdx = tg_CacheLoadDynIdx * NUM_DDR_LOAD_PER_LOOP + tg_regCacheLoadIdx;
                        //K stored as HWCK, threadgroup is loading 64 kernels at a time to LDS in a linear fashion.
                        //HW from tg_kernelSpatialOffset
                        //C from tg_ChannelOffset+tg_CacheLoadIdx
                        //K from tg_KernelBaseId (for TG) + threadIndex ([0-63])
                        uint tg_KernelReadOffset = tg_kernelSpatialOffset + tg_NumKernels*(tg_ChannelOffset+tg_CacheLoadIdx) + tg_KernelBaseId;
                        uint kernelReadOffset = tg_KernelReadOffset + threadIndex;
                        #if !STRICT_CHANNELS || LAX_KERNEL
                            kernelReadOffset = min(kernelReadOffset, K.GetLength()-1);
                        #endif
                        tempW[tg_regCacheLoadIdx] = K.FastGet(kernelReadOffset);

                        //Compute input position and mask.
                        #if SHUFFLE_FOR_COALESCED_LOAD
                            //64 Reads per TG per loop -> 4 pixels x 16 channels across threads -> good for NHWC.
                            //IMPORTANT : For register pressure reason -> it is assumed that tg_WidthO % 4 == 0, so we know all
                            //pixels for a given TG+loop are on the same row and thus we can compute Y mask/pos using scalar registers.
                            uint cacheChannelId = threadIndex % 16;
                            int tg_outputPixelBaseId = tg_OutputPixelBaseId + tg_CacheLoadIdx * 4;
                            int2 tg_ouputPixelsBaseCoord = int2(tg_outputPixelBaseId % tg_WidthO, tg_outputPixelBaseId / tg_WidthO);
                            int2 tg_inputPixelsBaseCoord = tg_ouputPixelsBaseCoord * _Stride.xy - _Pad.xy + int2(tg_Dx, tg_Dy);
                            bool tg_inputPixelsYMask = (tg_inputPixelsBaseCoord.y >= 0) && (tg_inputPixelsBaseCoord.y < (int)tg_HeightX);
                            int inputPixelXCoord = (threadIndex / 16) * _Stride.x + tg_inputPixelsBaseCoord.x;
                            bool inputPixelMask = tg_inputPixelsYMask && (inputPixelXCoord >= 0) && (inputPixelXCoord < (int)tg_WidthX);
                            int2 inputPixelCoords = int2(inputPixelXCoord, tg_inputPixelsBaseCoord.y);//.y is scalar
                        #else
                            //64 Reads per TG per loop -> 64 pixels across threads -> good for NCHW.
                            uint cacheChannelId = tg_CacheLoadIdx;//scalar in that code path.
                            int outputPixelBaseId = tg_OutputPixelBaseId + threadIndex;
                            int2 outputPixelCoords = int2(outputPixelBaseId % tg_WidthO, outputPixelBaseId / tg_WidthO);
                            int2 inputPixelCoords = outputPixelCoords * _Stride.xy - _Pad.xy + int2(tg_Dx, tg_Dy);
                            bool inputPixelMask = all( (inputPixelCoords >= 0) && (inputPixelCoords < float2(tg_WidthX, tg_HeightX)) );
                        #endif
                        int inputPixelId = inputPixelCoords.y * tg_WidthX + inputPixelCoords.x;
                        uint inputChannelId = tg_ChannelOffset + cacheChannelId;
                        bool inputChannelMask = inputChannelId < tg_NumChannels;
                        #if STRICT_CHANNELS
                            inputChannelMask = true;
                        #endif
                        #if CHANNELS_FIRST
                            uint pixelReadOffset = tg_NumInputPixels * inputChannelId + inputPixelId + tg_BatchReadOffset;
                        #else
                            uint pixelReadOffset = tg_NumChannels * inputPixelId + inputChannelId + tg_BatchReadOffset;
                        #endif
                        tempX[tg_regCacheLoadIdx] = X.MaskedGet(inputPixelMask && inputChannelMask, pixelReadOffset);
                    }

                    [unroll] for (tg_regCacheLoadIdx = 0; tg_regCacheLoadIdx < NUM_DDR_LOAD_PER_LOOP; ++tg_regCacheLoadIdx)
                    {
                        uint tg_CacheLoadIdx = tg_CacheLoadDynIdx * NUM_DDR_LOAD_PER_LOOP + tg_regCacheLoadIdx;
                        #if SHUFFLE_FOR_COALESCED_LOAD
                            uint cachePixelId = tg_CacheLoadIdx * 4 + threadIndex / 16;
                            uint cacheChannelId = threadIndex % 16;
                        #else
                            uint cachePixelId = threadIndex;
                            uint cacheChannelId = tg_CacheLoadIdx;//scalar in that code path.
                        #endif
                        uint weightWriteIndex = (threadIndex>31)?threadIndex+CACHE_WIDTH_W_PAD:threadIndex;
                        LDS_[ W_OFFSET + tg_CacheLoadIdx*CACHE_WIDTH_W + weightWriteIndex ] = tempW[tg_regCacheLoadIdx];
                        LDS_[ X_OFFSET + cacheChannelId*CACHE_WIDTH_X + cachePixelId ] = tempX[tg_regCacheLoadIdx];
                    }
                }

                GroupMemoryBarrierWithGroupSync();

                //Inner loop
                uint ptrX = groupThreadID.y*BLOCK_SIZE + X_OFFSET;
                uint ptrW = groupThreadID.x*BLOCK_SIZE + W_OFFSET;
                ptrW += (groupThreadID.x*BLOCK_SIZE>31)?CACHE_WIDTH_W_PAD:0;
                for (uint tg_CacheExecuteIdx = 0; tg_CacheExecuteIdx < CACHE_DEPTH; ++tg_CacheExecuteIdx)
                {
                    //Load LDS -> registers
                    float colOfX[BLOCK_SIZE];
                    float rowOfW[BLOCK_SIZE];
                    uint tg_q;
                    [unroll] for (tg_q = 0; tg_q < BLOCK_SIZE; ++tg_q)
                        colOfX[tg_q] = LDS_[ptrX + tg_q];
                    [unroll] for (tg_q = 0; tg_q < BLOCK_SIZE; ++tg_q)
                        rowOfW[tg_q] = LDS_[ptrW + tg_q];

                    ptrX += CACHE_WIDTH_X;
                    ptrW += CACHE_WIDTH_W;

                    //Mads 8 pixels by 8 kernels matmul style --> 64 mads
                    [unroll] for (uint tg_X = 0; tg_X < BLOCK_SIZE; ++tg_X)
                        [unroll] for (uint tg_W = 0; tg_W < BLOCK_SIZE; ++tg_W)
                            dstA[tg_X*BLOCK_SIZE+tg_W] = ffma(colOfX[tg_X], rowOfW[tg_W], dstA[tg_X*BLOCK_SIZE+tg_W]);
                }

                GroupMemoryBarrierWithGroupSync();
            }

            tg_kernelSpatialOffset += tg_KernelSpatialStride;
        }
    }

    #if SHUFFLE_FOR_COALESCED_STORE
        //-----------------------------------------------------
        //Use LDS to shuffle TG registers into coalesced writes
        //-----------------------------------------------------
        //A TG output [64pixels,64channels] = 4096 values, We will process [32,64] values at a time per TG.
        #if CHANNELS_FIRST
            //NCHW
            for (uint tg_registerChannelOffset = 0; tg_registerChannelOffset < BLOCK_SIZE; tg_registerChannelOffset += 4)
            {
                //Store 8 pixels x 4 channels per threads to LDS.
                [unroll] for (tg_kId = 0; tg_kId < 4; ++tg_kId)
                    [unroll] for (tg_pId = 0; tg_pId < BLOCK_SIZE; ++tg_pId)
                    {
                        //To avoid bank conflict store in 32 groups [8pixelsGroups,4channelsGroups] each group contain 64 values [8pixels,8kernels] for a total of 2048 values [64pixels,32channels]
                        uint ldsOffsetOfGroup = CACHE_WIDTH_X * (tg_kId*BLOCK_SIZE+tg_pId);//64 * ([0,3]*8+[0,7]) = [0,1984]
                        LDS_[ldsOffsetOfGroup + threadIndex] = dstA[BLOCK_SIZE * tg_pId + (tg_registerChannelOffset + tg_kId)];
                    }

                GroupMemoryBarrierWithGroupSync();

                //We have a buffers of [64pixels,32channels] floats, each thread will store [1pixels,32channels] so a threadgroup is storing 64 pixels at a time to DDR in a linear fashion.
                uint readPixelId = threadIndex;
                uint writePixelId = tg_OutputPixelBaseId + readPixelId;

                #define WRITE_8CHANNELS_IF_POSSIBLE(groupID) \
                tg_ddrChannelGroupBaseId[groupID] = tg_KernelBaseId + 16 * groupID; \
                if (tg_ddrChannelGroupBaseId[groupID] < tg_NumKernels) \
                { \
                    [unroll] for (tg_kId = groupID*8; tg_kId < 8*(groupID+1); ++tg_kId) \
                    { \
                        uint tg_kIdOfGroup = tg_kId % 4; \
                        uint pIdOfGroup = readPixelId % BLOCK_SIZE; \
                        uint ldsOffsetOfGroup = CACHE_WIDTH_X * (tg_kIdOfGroup * BLOCK_SIZE + pIdOfGroup); \
                        uint tg_kIdInGroup = (tg_kId - tg_kIdOfGroup) / 4; \
                        uint pIdInGroup = (readPixelId - pIdOfGroup) / BLOCK_SIZE; \
                        uint ldsOffsetInGroup = pIdInGroup * BLOCK_SIZE + tg_kIdInGroup; \
                        uint readIndex = ldsOffsetOfGroup + ldsOffsetInGroup; \
                        uint writeChannelId = tg_KernelBaseId + tg_kId%4 + (tg_kId/4)*BLOCK_SIZE + tg_registerChannelOffset; \
                        uint writeIndex = O.width * O.height * writeChannelId + writePixelId + tg_BatchWriteOffset; \
                        O.FastSetWithActivation(writeIndex, LDS_[readIndex]); \
                    } \
                }

                if (writePixelId < tg_NumOuputPixels)
                {
                    #if LAX_KERNEL
                        uint tg_ddrChannelGroupBaseId[4];
                        WRITE_8CHANNELS_IF_POSSIBLE(0);
                        WRITE_8CHANNELS_IF_POSSIBLE(1);
                        WRITE_8CHANNELS_IF_POSSIBLE(2);
                        WRITE_8CHANNELS_IF_POSSIBLE(3);
                    #else
                        [unroll] for (tg_kId = 0; tg_kId < 32; ++tg_kId)
                        {
                            //Find LDS group to read from
                            uint tg_kIdOfGroup = tg_kId % 4;//[0,3] kernelsGroups
                            uint pIdOfGroup = readPixelId % BLOCK_SIZE;//[0,7] pixelsGroups
                            uint ldsOffsetOfGroup = CACHE_WIDTH_X * (tg_kIdOfGroup * BLOCK_SIZE + pIdOfGroup);//CACHE_WIDTH_X * ([0,3]*8+[0,7]) = [0,1984]
                            //Find index inside that group
                            uint tg_kIdInGroup = (tg_kId - tg_kIdOfGroup) / 4;//[0,7] kernels
                            uint pIdInGroup = (readPixelId - pIdOfGroup) / BLOCK_SIZE;//[0,7] pixels
                            uint ldsOffsetInGroup = pIdInGroup * BLOCK_SIZE + tg_kIdInGroup;//[0,7]*8+[0,7] = [0,63]
                            //load from LDS and store to DDR
                            uint readIndex = ldsOffsetOfGroup + ldsOffsetInGroup;//[0,2047]
                            uint writeChannelId = tg_KernelBaseId + tg_kId%4 + (tg_kId/4)*BLOCK_SIZE + tg_registerChannelOffset;
                            uint writeIndex = O.width * O.height * writeChannelId + writePixelId + tg_BatchWriteOffset;
                            //TODO Still some bank conflict here, an option would be to pad LDS but need more loop then (as already have 8k LDS with two loop).
                            O.FastSetWithActivation(writeIndex, LDS_[readIndex]);
                        }
                    #endif
                }

                GroupMemoryBarrierWithGroupSync();
            }
        #else
            //NHWC
            for (uint tg_registerPixelOffset = 0; tg_registerPixelOffset < BLOCK_SIZE; tg_registerPixelOffset += 4)
            {
                //Store 4 pixels x 8 channels per threads to LDS.
                uint ldsRowOffset = groupThreadID.y * 4;
                uint ldsChannelOffset = groupThreadID.x * BLOCK_SIZE;
                [unroll] for (tg_pId = 0; tg_pId < 4; ++tg_pId)
                    [unroll] for (tg_kId = 0; tg_kId < BLOCK_SIZE; ++tg_kId)
                    {
                        //TODO check for bank conflict here, probably need to swizzle the writes per thread
                        LDS_[CACHE_WIDTH_X * (ldsRowOffset + tg_pId) + ldsChannelOffset + tg_kId] = dstA[BLOCK_SIZE * (tg_registerPixelOffset + tg_pId) + tg_kId];
                    }

                GroupMemoryBarrierWithGroupSync();

                //We have a buffers of [32pixels,64channels] floats, each thread will store [32pixels,1channels] so a threadgroup is storing 64 kernels at a time to DDR in a linear fashion.
                uint writeChannelId = tg_KernelBaseId + threadIndex;
                uint tg_writeLoopBaseId = tg_OutputPixelBaseId + tg_registerPixelOffset;
                uint tg_ddrPixelGroupBaseId[8];

                #if LAX_KERNEL
                    bool canWriteChannel = (writeChannelId < tg_NumKernels);
                #else
                    bool canWriteChannel = true;
                #endif

                //Ok as we enforce W%4==0 thus W*H%4==0 also.
                //Using a Macro as [unroll] on loop(groupID) won't unroll properly and thus introduce LDS/DDR sync points.
                #define WRITE_4PIXELS_IF_POSSIBLE(groupID) \
                tg_ddrPixelGroupBaseId[groupID]= tg_writeLoopBaseId + BLOCK_SIZE * groupID; \
                if ((tg_ddrPixelGroupBaseId[groupID] < tg_NumOuputPixels) && canWriteChannel)\
                { \
                    [unroll] for (tg_pId = 0; tg_pId < 4; ++tg_pId) \
                        O.FastSetWithActivation(tg_BatchWriteOffset + tg_NumKernels * (tg_ddrPixelGroupBaseId[groupID]+tg_pId) + writeChannelId, LDS_[CACHE_WIDTH_X * (groupID * 4 + tg_pId) + threadIndex]); \
                }
                WRITE_4PIXELS_IF_POSSIBLE(0);
                WRITE_4PIXELS_IF_POSSIBLE(1);
                WRITE_4PIXELS_IF_POSSIBLE(2);
                WRITE_4PIXELS_IF_POSSIBLE(3);
                WRITE_4PIXELS_IF_POSSIBLE(4);
                WRITE_4PIXELS_IF_POSSIBLE(5);
                WRITE_4PIXELS_IF_POSSIBLE(6);
                WRITE_4PIXELS_IF_POSSIBLE(7);
                #undef WRITE_PIXEL_GROUP_IF_POSSIBLE

                GroupMemoryBarrierWithGroupSync();
            }
        #endif //CHANNELS_FIRST
    #else
		//-------------------------------
		//Directly store registers to DDR
		//-------------------------------
		//B does not require an offset as size == 1
		//C from tg_KernelBaseId, groupThreadID.x and tg_kId
		//HW from tg_OutputPixelBaseId, groupThreadID.y and tg_pId
        [unroll] for (tg_kId = 0; tg_kId < BLOCK_SIZE; ++tg_kId)
            [unroll] for (tg_pId = 0; tg_pId < BLOCK_SIZE; ++tg_pId)
            {
                uint writeChannelId = tg_KernelBaseId + groupThreadID.x * BLOCK_SIZE + tg_kId;
                uint writePixelId = tg_OutputPixelBaseId + groupThreadID.y * BLOCK_SIZE + tg_pId;
                float writeValue = dstA[tg_pId*BLOCK_SIZE+tg_kId];
                #if CHANNELS_FIRST
                    uint writeIndex = O.width * O.height * writeChannelId + writePixelId + tg_BatchWriteOffset;
                #else
                    uint writeIndex = tg_NumKernels * writePixelId + writeChannelId + tg_BatchWriteOffset;
                #endif
                #if LAX_KERNEL
                    bool canWriteChannel = (writeChannelId < tg_NumKernels);
                #else
                    bool canWriteChannel = true;
                #endif
                if ((writePixelId < tg_NumOuputPixels) && canWriteChannel)
                    O.FastSetWithActivation(writeIndex, writeValue);
            }
    #endif

    #undef X_OFFSET
    #undef W_OFFSET
    #undef LDS_
    #undef X_
    #undef W_
}
#undef CACHE_DEPTH
#undef CACHE_WIDTH
#undef SHUFFLE_FOR_COALESCED_LOAD
#undef SHUFFLE_FOR_COALESCED_STORE
#endif //KERNEL_PER_TG == 64

#if KERNEL_PER_TG == 16

#define CACHE_DEPTH 4                          // This kernel code supports only CACHE_DEPTH=4, this value can not be changed
#define PIXELS_PER_CACHE 256                   // This kernel code supports only PIXELS_PER_CACHE=256, this value can not be changed
#define NUMTHREADS_PER_TG 64                   // This kernel code supports only NUMTHREADS_PER_TG=64, this value can not be changed
#define PIXELS_READ_PER_THREAD_PER_CACHE       PIXELS_PER_CACHE/NUMTHREADS_PER_TG

#if CHANNELS_FIRST
    //NCHW
    #define PIXELS_CACHE_PAD 1
    #define SHUFFLE_FOR_COALESCED_LOAD 0       // Not needed for NCHW
    #define SHUFFLE_FOR_COALESCED_STORE 1
#else
    //NHWC
    #define PIXELS_CACHE_PAD 0                 // TODO not implemented for NHWC
    #define SHUFFLE_FOR_COALESCED_LOAD 1
    #define SHUFFLE_FOR_COALESCED_STORE 0      // Not implemented for NHWC, TODO (probably limited gain because of CACHE_DEPTH of 4)
#endif

#define PIXELS_PER_CACHE_AND_PAD ((PIXELS_PER_CACHE/BLOCK_SIZE)*(BLOCK_SIZE+PIXELS_CACHE_PAD))

#if SHUFFLE_FOR_COALESCED_STORE
    //A TG output [256pixels,16channels] = 4096 values, We will write two time 2048 values to DDR (8k LDS).
    groupshared float CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, LDS)[2048];
#else
    groupshared float CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, LDS)[(KERNEL_PER_TG+PIXELS_PER_CACHE_AND_PAD)*CACHE_DEPTH];
#endif
[numthreads(2,32,1)]
void FUNC_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE)(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID, uint threadIndex : SV_GroupIndex)
{
    //This kernel assume the following:
    //Input:
    //  C % CACHE_DEPTH==0 <-- only if STRICT_CHANNELS==1
    //Kernel:
    //  K%16==0 <-- only if LAX_KERNEL=0
    //DISPATCH ARGS(K.kernelCount, O.width * O.height, O.batch);
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);
    #define LDS_ CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, LDS)
    #define X_OFFSET 0
    #define W_OFFSET CACHE_DEPTH*PIXELS_PER_CACHE_AND_PAD

    //Per thread group (scalar registers)
    uint tg_NumChannels = X.channels;
    uint tg_WidthX  = X.width;
    uint tg_HeightX = X.height;
    uint tg_WidthO  = O.width;
    uint tg_HeightO = O.height;
    uint tg_NumKernels = K.channels;
    uint tg_NumInputPixels = tg_WidthX*tg_HeightX;
    uint tg_NumOuputPixels = tg_WidthO*tg_HeightO;
    uint tg_KernelSpatialStride = tg_NumKernels*tg_NumChannels;
    uint tg_KernelBaseId = groupID.x * KERNEL_PER_TG;
    uint tg_OutputPixelBaseId = groupID.y * PIXELS_PER_CACHE;
    uint tg_BatchReadOffset = groupID.z * tg_NumChannels * tg_HeightX * tg_WidthX;
    uint tg_BatchWriteOffset = groupID.z * tg_NumKernels * tg_HeightO * tg_WidthO;
    uint tg_kernelSpatialOffset = 0;

    //8x8 block, 8 kernels by 8 pixels
    //**********************************
    //* Kernel Ids  *  0  1  2  3  ...
    //**********************************
    //              *  ThreadIds
    // Pixel Ids  0 *  0  1  2  3 ...
    //            1 *  8  9 10 11 ...
    //            2 * 16 17 18 19 ...
    //            3 * 32 33 34 35 ...
    //            ... ...
    float dstA[BLOCK_SIZE*BLOCK_SIZE];

    //Load Bias [K] int dstA [Kernels, Pixels]
    uint tg_kId;
    uint tg_pId;
    uint maxBiasIndex = O.channels - 1;
    [unroll] for (tg_pId = 0; tg_pId < BLOCK_SIZE; ++tg_pId)
        [unroll] for (tg_kId = 0; tg_kId < BLOCK_SIZE; ++tg_kId)
            dstA[tg_pId*BLOCK_SIZE+tg_kId] = B.FastGet(min(maxBiasIndex,tg_KernelBaseId + groupThreadID.x * BLOCK_SIZE + tg_kId));

    //Loop spatialy on kernels
    for (uint tg_Dy = 0; tg_Dy < K.GetKernelHeight(); tg_Dy++)
    {
        for (uint tg_Dx = 0; tg_Dx < K.GetKernelWidth(); tg_Dx++)
        {
            for (uint tg_ChannelOffset = 0; tg_ChannelOffset < tg_NumChannels; tg_ChannelOffset += CACHE_DEPTH)
            {
                //Load from DDR to LDS: (16*CACHE_DEPTH=64 weights + 256*CACHE_DEPTH=1024 pixels) => 4352Bytes * CACHE_DEPTH.

                //K stored as HWCK, threadgroup is loading 64 kernels at a time to LDS in a linear fashion (4x16 kernels).
                //HW from tg_kernelSpatialOffset
                //C from tg_ChannelOffset (for TG) + threadIndex ([0-63]->[0-3])
                //K from tg_KernelBaseId (for TG) + threadIndex ([0-63])
                uint kernelCacheLoadOffset = threadIndex / 16;
                uint kernelLoadOffset = threadIndex % 16;
                uint kernelReadOffset = tg_kernelSpatialOffset + tg_NumKernels*(tg_ChannelOffset+kernelCacheLoadOffset) + tg_KernelBaseId + kernelLoadOffset;
                #if !STRICT_CHANNELS || LAX_KERNEL
                    kernelReadOffset = min(kernelReadOffset, K.GetLength()-1);
                #endif
                float tempW = K.FastGet(kernelReadOffset);

                #if SHUFFLE_FOR_COALESCED_LOAD
                    //Good for HWC
                    //TG is loading 256Pixels * CACHE_DEPTH to LDS in an attempt of linear fashion (16 pixels read per thread).
                    //would be better if CACHE_DEPTH would be bigger than 4 but LDS is the limiting factor here.
                    uint tg_PixelLoadIdx;
                    uint cacheLoadIdx = threadIndex % 4;
                    uint pixelLoadOffset = threadIndex / 4;
                    float tempX[CACHE_DEPTH*PIXELS_READ_PER_THREAD_PER_CACHE];//{channels*pixels}
                    [unroll] for (tg_PixelLoadIdx = 0; tg_PixelLoadIdx < PIXELS_READ_PER_THREAD_PER_CACHE*CACHE_DEPTH; ++tg_PixelLoadIdx)
                    {
                        //Compute input position and mask.
                        int outputPixelBaseId = tg_OutputPixelBaseId + PIXELS_READ_PER_THREAD_PER_CACHE*CACHE_DEPTH * tg_PixelLoadIdx + pixelLoadOffset;
                        int2 outputPixelCoords = int2(outputPixelBaseId % tg_WidthO, outputPixelBaseId / tg_WidthO);
                        int2 inputPixelCoords = outputPixelCoords * _Stride.xy - _Pad.xy + int2(tg_Dx, tg_Dy);
                        bool inputPixelMask = all( (inputPixelCoords >= 0) && (inputPixelCoords < float2(tg_WidthX, tg_HeightX)) );

                        int inputPixelId = inputPixelCoords.y * tg_WidthX + inputPixelCoords.x;
                        uint tg_InputChannelId = tg_ChannelOffset + cacheLoadIdx;
                        bool inputChannelMask = tg_InputChannelId < tg_NumChannels;
                        #if STRICT_CHANNELS
                            inputChannelMask = true;
                        #endif
                        #if CHANNELS_FIRST
                            uint pixelReadOffset = tg_NumInputPixels * tg_InputChannelId + inputPixelId + tg_BatchReadOffset;
                        #else
                            uint pixelReadOffset = tg_NumChannels * inputPixelId + tg_InputChannelId + tg_BatchReadOffset;
                        #endif
                        tempX[tg_PixelLoadIdx] = X.MaskedGet(inputPixelMask && inputChannelMask, pixelReadOffset);
                    }

                    [unroll] for (tg_PixelLoadIdx = 0; tg_PixelLoadIdx < PIXELS_READ_PER_THREAD_PER_CACHE*CACHE_DEPTH; ++tg_PixelLoadIdx)
                    {
                        LDS_[ X_OFFSET + cacheLoadIdx*PIXELS_PER_CACHE_AND_PAD + tg_PixelLoadIdx*PIXELS_READ_PER_THREAD_PER_CACHE*CACHE_DEPTH + pixelLoadOffset] = tempX[tg_PixelLoadIdx];
                    }
                #else
                    //Good for CHW
                    //TG is loading 256Pixels * CACHE_DEPTH to LDS in a linear fashion (4 channels * 4 pixels read per thread).
                    //Explicit register declaration as [unroll] won't unroll properly otherwise and introduce sync points.
                    uint tg_CacheLoadIdx;
                    uint tg_PixelLoadIdx;
                    float tempX[CACHE_DEPTH][PIXELS_READ_PER_THREAD_PER_CACHE];//{channels,pixels}
                    [unroll] for (tg_CacheLoadIdx = 0; tg_CacheLoadIdx < CACHE_DEPTH; ++tg_CacheLoadIdx)
                    {
                        [unroll] for (tg_PixelLoadIdx = 0; tg_PixelLoadIdx < PIXELS_READ_PER_THREAD_PER_CACHE; ++tg_PixelLoadIdx)
                        {
                            //Compute input position and mask.
                            int outputPixelBaseId = tg_OutputPixelBaseId + NUMTHREADS_PER_TG * tg_PixelLoadIdx + threadIndex;
                            int2 outputPixelCoords = int2(outputPixelBaseId % tg_WidthO, outputPixelBaseId / tg_WidthO);
                            int2 inputPixelCoords = outputPixelCoords * _Stride.xy - _Pad.xy + int2(tg_Dx, tg_Dy);
                            bool inputPixelMask = all( (inputPixelCoords >= 0) && (inputPixelCoords < float2(tg_WidthX, tg_HeightX)) );

                            int inputPixelId = inputPixelCoords.y * tg_WidthX + inputPixelCoords.x;
                            uint tg_InputChannelId = tg_ChannelOffset + tg_CacheLoadIdx;
                            bool inputChannelMask = tg_InputChannelId < tg_NumChannels;
                            #if STRICT_CHANNELS
                                inputChannelMask = true;
                            #endif
                            #if CHANNELS_FIRST
                                uint pixelReadOffset = tg_NumInputPixels * tg_InputChannelId + inputPixelId + tg_BatchReadOffset;
                            #else
                                uint pixelReadOffset = tg_NumChannels * inputPixelId + tg_InputChannelId + tg_BatchReadOffset;
                            #endif
                            tempX[tg_CacheLoadIdx][tg_PixelLoadIdx] = X.MaskedGet(inputPixelMask && inputChannelMask, pixelReadOffset);
                        }
                    }

                    [unroll] for (tg_CacheLoadIdx = 0; tg_CacheLoadIdx < CACHE_DEPTH; ++tg_CacheLoadIdx)
                    {
                        [unroll] for (tg_PixelLoadIdx = 0; tg_PixelLoadIdx < PIXELS_READ_PER_THREAD_PER_CACHE; ++tg_PixelLoadIdx)
                        {
                            uint ldsPixelCacheWriteIndex = tg_PixelLoadIdx*NUMTHREADS_PER_TG + threadIndex;
                            ldsPixelCacheWriteIndex += (ldsPixelCacheWriteIndex/BLOCK_SIZE) * PIXELS_CACHE_PAD;
                            LDS_[ X_OFFSET + tg_CacheLoadIdx*PIXELS_PER_CACHE_AND_PAD + ldsPixelCacheWriteIndex] = tempX[tg_CacheLoadIdx][tg_PixelLoadIdx];
                        }
                    }
                #endif
                LDS_[ W_OFFSET + kernelCacheLoadOffset*KERNEL_PER_TG + kernelLoadOffset ] = tempW;

                GroupMemoryBarrierWithGroupSync();

                //Inner loop
                uint ptrX = groupThreadID.y*(BLOCK_SIZE+PIXELS_CACHE_PAD) + X_OFFSET;
                uint ptrW = groupThreadID.x*BLOCK_SIZE + W_OFFSET;
                for (uint tg_CacheExecuteIdx = 0; tg_CacheExecuteIdx < CACHE_DEPTH; ++tg_CacheExecuteIdx)
                {
                    //Load LDS -> registers
                    float colOfX[BLOCK_SIZE];
                    float rowOfW[BLOCK_SIZE];
                    uint tg_q;
                    [unroll] for (tg_q = 0; tg_q < BLOCK_SIZE; ++tg_q)
                        colOfX[tg_q] = LDS_[ptrX + tg_q];
                    [unroll] for (tg_q = 0; tg_q < BLOCK_SIZE; ++tg_q)
                        rowOfW[tg_q] = LDS_[ptrW + tg_q];

                    ptrX += PIXELS_PER_CACHE_AND_PAD;
                    ptrW += KERNEL_PER_TG;

                    //Mads 8 pixels by 8 kernels matmul style --> 64 mads
                    [unroll] for (uint tg_X = 0; tg_X < BLOCK_SIZE; ++tg_X)
                        [unroll] for (uint tg_W = 0; tg_W < BLOCK_SIZE; ++tg_W)
                            dstA[tg_X*BLOCK_SIZE+tg_W] = ffma(colOfX[tg_X], rowOfW[tg_W], dstA[tg_X*BLOCK_SIZE+tg_W]);
                }

                GroupMemoryBarrierWithGroupSync();
            }

            tg_kernelSpatialOffset += tg_KernelSpatialStride;
        }
    }

    #if SHUFFLE_FOR_COALESCED_STORE && !LAX_KERNEL
        //-----------------------------------------------------
        //Use LDS to shuffle TG registers into coalesced writes
        //-----------------------------------------------------
        //A TG output [256pixels,16channels] = 4096 values, We will process [256,8] values at a time per TG.
        for (uint tg_registerChannelOffset = 0; tg_registerChannelOffset < BLOCK_SIZE; tg_registerChannelOffset += 4)
        {
            //Store 8 pixels x 4 channels per threads to LDS.
            [unroll] for (tg_kId = 0; tg_kId < 4; ++tg_kId)
                [unroll] for (tg_pId = 0; tg_pId < BLOCK_SIZE; ++tg_pId)
                {
                    //To avoid bank conflict store in 32 groups [8pixelsGroups,4channelsGroups] each group contain 64 values [32pixels,2kernels] for a total of 2048 values [256pixels,8channels]
                    uint ldsOffsetOfGroup = NUMTHREADS_PER_TG * (tg_kId*BLOCK_SIZE+tg_pId);//64 * ([0,3]*8+[0,7]) = [0,1984]
                    LDS_[ldsOffsetOfGroup + threadIndex] = dstA[BLOCK_SIZE * tg_pId + (tg_registerChannelOffset + tg_kId)];
                }

            GroupMemoryBarrierWithGroupSync();

            //We have a buffers of [256pixels,8channels] floats, each thread will store [4pixels,8channels] so a threadgroup is storing 64 pixels at a time to DDR in a linear fashion.
            //Using a Macro as [unroll] on loop(groupID) won't unroll properly and thus introduce LDS/DDR sync points.
            #define WRITE_8CHANNELS_IF_POSSIBLE(groupID) \
            {\
                uint readPixelId = groupID * NUMTHREADS_PER_TG + threadIndex; \
                uint writePixelId = tg_OutputPixelBaseId + groupID * NUMTHREADS_PER_TG + threadIndex; \
                if (writePixelId < tg_NumOuputPixels) \
                { \
                    [unroll] for (tg_kId = 0; tg_kId < BLOCK_SIZE; ++tg_kId) \
                    { \
                        uint tg_kIdOfGroup = tg_kId % 4; \
                        uint pIdOfGroup = readPixelId % BLOCK_SIZE; \
                        uint ldsOffsetOfGroup = NUMTHREADS_PER_TG * (tg_kIdOfGroup * BLOCK_SIZE + pIdOfGroup); \
                        uint tg_kIdInGroup = (tg_kId - tg_kIdOfGroup) / 4; \
                        uint pIdInGroup = (readPixelId - pIdOfGroup) / BLOCK_SIZE; \
                        uint ldsOffsetInGroup = pIdInGroup * 2 + tg_kIdInGroup; \
                        uint readIndex = ldsOffsetOfGroup + ldsOffsetInGroup; \
                        uint writeChannelId = tg_KernelBaseId + tg_kId%4 + (tg_kId/4)*BLOCK_SIZE + tg_registerChannelOffset; \
                        uint writeIndex = O.width * O.height * writeChannelId + writePixelId + tg_BatchWriteOffset; \
                        O.FastSetWithActivation(writeIndex, LDS_[readIndex]); \
                    } \
                } \
            }
            WRITE_8CHANNELS_IF_POSSIBLE(0)
            WRITE_8CHANNELS_IF_POSSIBLE(1)
            WRITE_8CHANNELS_IF_POSSIBLE(2)
            WRITE_8CHANNELS_IF_POSSIBLE(3)
            #undef WRITE_8CHANNELS_IF_POSSIBLE

            GroupMemoryBarrierWithGroupSync();
        }
    #else
        //-------------------------------
        //Directly store registers to DDR
        //-------------------------------
        //B does not require an offset as size == 1
        //C from tg_KernelBaseId, groupThreadID.x and tg_kId
        //HW from tg_OutputPixelBaseId, groupThreadID.y and tg_pId
        [unroll] for (tg_kId = 0; tg_kId < BLOCK_SIZE; ++tg_kId)
            [unroll] for (tg_pId = 0; tg_pId < BLOCK_SIZE; ++tg_pId)
            {
                uint writeChannelId = tg_KernelBaseId + groupThreadID.x * BLOCK_SIZE + tg_kId;
                uint writePixelId = tg_OutputPixelBaseId + groupThreadID.y * BLOCK_SIZE + tg_pId;
                float writeValue = dstA[tg_pId*BLOCK_SIZE+tg_kId];
                #if CHANNELS_FIRST
                    uint writeIndex = O.width * O.height * writeChannelId + writePixelId + tg_BatchWriteOffset;
                #else
                    uint writeIndex = tg_NumKernels * writePixelId + writeChannelId + tg_BatchWriteOffset;
                #endif
                #if LAX_KERNEL
                    bool canWriteChannel = (writeChannelId < tg_NumKernels);
                #else
                    bool canWriteChannel = true;
                #endif
                if ((writePixelId < tg_NumOuputPixels) && canWriteChannel)
                    O.FastSetWithActivation(writeIndex, writeValue);
            }
    #endif

    #undef X_OFFSET
    #undef W_OFFSET
    #undef LDS_
    #undef X_
    #undef W_
}
#undef CACHE_DEPTH
#undef PIXELS_READ_PER_THREAD_PER_CACHE
#undef PIXELS_PER_CACHE
#undef NUMTHREADS_PER_TG
#undef SHUFFLE_FOR_COALESCED_LOAD
#undef SHUFFLE_FOR_COALESCED_STORE
#endif //KERNEL_PER_TG == 16

#endif //BLOCK_SIZE == 8

#if BLOCK_SIZE == 4
#define BUF_OFFSET 0
#define CACHE_DEPTH 16 // This kernel code supports only CACHE_DEPTH=16, this value can not be changed
#define SHUFFLE_FOR_COALESCED_STORE 1 // Only implemented in CHW path.
groupshared float CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, LDS)[2*CACHE_DEPTH*16*BLOCK_SIZE+(1-CHANNELS_FIRST)*CACHE_DEPTH];
[numthreads(16,16,1)]
void FUNC_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE)(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint threadIndex : SV_GroupIndex)
{
    //DISPATCH ARGS(K.kernelCount, O.width * O.height * O.batch, 1); // in NHWC
    //DISPATCH ARGS(K.kernelCount, O.width * O.height, O.batch);     // in NCHW

    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);

    // [W*H, Ky*Kx*In] * [Ky*Kx*In, Out] => [W*H, Out]
    #define LDS_ CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, LDS)
    #define X_OFFSET 0
    #define W_OFFSET CACHE_DEPTH*16*BLOCK_SIZE+(1-CHANNELS_FIRST)*CACHE_DEPTH

    int x = (int)dispatchThreadID.x * BLOCK_SIZE; // output_channels
    int y = (int)dispatchThreadID.y * BLOCK_SIZE; // batch*width*height (width*height in HWC)
    int tx = (int)groupThreadID.x;
    int ty = (int)groupThreadID.y;
    int bx = ((int)dispatchThreadID.x - (int)groupThreadID.x) * BLOCK_SIZE;
    int by = ((int)dispatchThreadID.y - (int)groupThreadID.y) * BLOCK_SIZE;
    int ti = (int)threadIndex;
    uint w      = O.width;
    uint h      = O.height;
    int batches = X.batch;
    int channels = X.channels;
    int widthX  = X.width;
    int heightX = X.height;
    int strideX = X.channels;
    int strideK = K.channels;
    int strideO = O.channels;
    int offsetX = BUF_OFFSET;
    int offsetK = BUF_OFFSET;
    int offsetO = BUF_OFFSET;
 #if CHANNELS_FIRST
    uint batchReadOffset = dispatchThreadID.z * channels * heightX * widthX;
    uint batchWriteOffset = dispatchThreadID.z * strideO * h * w;
    uint3 groupID = (dispatchThreadID - groupThreadID) / uint3(16,16,1);
    uint kernelBaseId = groupID.x * 64;
    uint outputPixelBaseId = groupID.y * 64;
    uint numOuputPixels = w * h;
 #endif

    float4 dstA[4];
    int maxBiasIndex = O.channels - 1;
    dstA[0].x = B.FastGet(min(maxBiasIndex, x+0)); dstA[0].y = B.FastGet(min(maxBiasIndex, x+1)); dstA[0].z = B.FastGet(min(maxBiasIndex, x+2)); dstA[0].w = B.FastGet(min(maxBiasIndex,x+3));
    dstA[1].x = B.FastGet(min(maxBiasIndex, x+0)); dstA[1].y = B.FastGet(min(maxBiasIndex, x+1)); dstA[1].z = B.FastGet(min(maxBiasIndex, x+2)); dstA[1].w = B.FastGet(min(maxBiasIndex,x+3));
    dstA[2].x = B.FastGet(min(maxBiasIndex, x+0)); dstA[2].y = B.FastGet(min(maxBiasIndex, x+1)); dstA[2].z = B.FastGet(min(maxBiasIndex, x+2)); dstA[2].w = B.FastGet(min(maxBiasIndex,x+3));
    dstA[3].x = B.FastGet(min(maxBiasIndex, x+0)); dstA[3].y = B.FastGet(min(maxBiasIndex, x+1)); dstA[3].z = B.FastGet(min(maxBiasIndex, x+2)); dstA[3].w = B.FastGet(min(maxBiasIndex,x+3));

    int readK = strideK * (ti>>6) + bx + (ti&63) + offsetK;
    #if STRICT_CHANNELS
    #else
    bool maskK = (bx + (ti&63)) < strideK;
    #endif

#if CHANNELS_FIRST
    uint centroidId = by + (ti&63);
    #if KERNEL_1x1
    int readX = heightX * widthX * (ti>>6) + centroidId + batchReadOffset;
    bool mask = centroidId < uint(widthX * heightX);
    #else
    int batch = 0;//not needed dispatched over batches.
    int topY = (centroidId / w % h) * _Stride.y - _Pad.y;
    int leftX = (centroidId % w) * _Stride.x - _Pad.x;
    int cornerId = batch * heightX * widthX + topY * widthX + leftX;
    int readX = heightX * widthX * (ti>>6) + cornerId + batchReadOffset;
    bool mask;
    #endif
#else
    uint4 centroidId = uint4(
        (by + (ti>>4) +  0),
        (by + (ti>>4) + 16),
        (by + (ti>>4) + 32),
        (by + (ti>>4) + 48));
    #if KERNEL_1x1
    int4 readX = strideX * centroidId + (ti&15);
    bool4 mask = centroidId < uint(batches * widthX * heightX);
    #else
    int4 batch = centroidId / w / h;
    int4 topY = (centroidId / w % h) * _Stride.y - _Pad.y;
    int4 leftX = (centroidId % w) * _Stride.x - _Pad.x;
    int4 cornerId = batch * heightX * widthX + topY * widthX + leftX;
    int4 readX = strideX * cornerId + (ti&15);
    bool4 mask;
    #endif
#endif

#if KERNEL_1x1
    {
        {
#else
    for (int dy = 0; dy < (int)K.GetKernelHeight(); dy++)
    {
        for (int dx = 0; dx < (int)K.GetKernelWidth(); dx++)
        {
            #if CHANNELS_FIRST
            int kernelOffsetX = (dy * widthX + dx);
            #else
            int kernelOffsetX = (dy * widthX + dx) * strideX;
            #endif
            mask =
                batch < batches &&
                topY + dy >= 0 &&
                topY + dy < heightX &&
                leftX + dx >= 0 &&
                leftX + dx < widthX;

            // 256 threads (256=numthreads(16,16,1)=16*16*1) are communally loading
            // blocks of 64pixels x 16channels from the global memory
            //
            // One block is read from X and one from K tensor
            // 4 reads with 256 threads (4=64*16/256) are necessary for each block

#endif // KERNEL_1x1
            for (int i = 0; i < channels; i += CACHE_DEPTH)
            {
                #if STRICT_CHANNELS
                #else
                if (i + CACHE_DEPTH > channels)
                {
                    int channelRemainder = channels - i;
                    [unroll] for (int j = 0; j < 4; ++j)
                    {
                        bool maskChannelsK = ti < 64 * (channelRemainder - j * 4);
                        bool maskChannelsX =
                            #if CHANNELS_FIRST
                            maskChannelsK;
                            #else
                            (ti&15) < channelRemainder;
                            #endif

                        LDS_[W_OFFSET + ((ti>>6)<<6) + ((ti&3)<<4) + ((ti&63)>>2) + 256*j] = K.MaskedGet(maskK & maskChannelsK, readK);
                        readK += strideK * max(0, min(channelRemainder - j * 4, 4));

                        #if CHANNELS_FIRST
                        LDS_[X_OFFSET + ti + 256*j] =
                            #if KERNEL_1x1
                            X.MaskedGet(mask && maskChannelsX, readX + heightX * widthX * (i + j * 4) + offsetX);
                            #else
                            X.MaskedGet(mask && maskChannelsX, readX + heightX * widthX * (i + j * 4) + kernelOffsetX + offsetX);
                            #endif
                        #else
                        LDS_[X_OFFSET + (ti>>4) + 65*(ti&15) + 16*j] =
                            #if KERNEL_1x1
                            X.MaskedGet(mask[j] && maskChannelsX, readX[j] + i + offsetX);
                            #else
                            X.MaskedGet(mask[j] && maskChannelsX, readX[j] + i + kernelOffsetX + offsetX);
                            #endif
                        #endif
                    }
                }
                else
                #endif
                [unroll] for (int j = 0; j < 4; ++j)
                {
                    LDS_[W_OFFSET + ((ti>>6)<<6) + ((ti&3)<<4) + ((ti&63)>>2) + 256*j] =
                        #if STRICT_CHANNELS
                        K.data[readK];
                        #else
                        K.MaskedGet(maskK, readK);
                        #endif
                    readK += strideK * 4;

                    #if CHANNELS_FIRST
                    LDS_[X_OFFSET + ti + 256*j] =
                        #if KERNEL_1x1
                        X.MaskedGet(mask, readX + heightX * widthX * (i + j * 4) + offsetX);
                        #else
                        X.MaskedGet(mask, readX + heightX * widthX * (i + j * 4) + kernelOffsetX + offsetX);
                        #endif
                    #else
                    LDS_[X_OFFSET + (ti>>4) + 65*(ti&15) + 16*j] =
                        #if KERNEL_1x1
                        X.MaskedGet(mask[j], readX[j] + i + offsetX);
                        #else
                        X.MaskedGet(mask[j], readX[j] + i + kernelOffsetX + offsetX);
                        #endif
                    #endif

                    #if DEBUG_CHECK_BOUNDS
                    if (
                        #if KERNEL_1x1
                        (readX[j] + i + offsetX < 0) ||
                        (readX[j] + i + offsetX >= (int)X.GetLength())
                        #else
                        (mask[j] && readX[j] + i + kernelOffsetX + offsetX < 0) ||
                        (mask[j] && readX[j] + i + kernelOffsetX + offsetX >= (int)X.GetLength())
                        #endif
                        )
                    {
                        // swamp X cache with dummy values when reading out of buffer
                        // this way we can detect out of buffer reads by comparing results from this kernel
                        // with the the reference implementation results
                        for (int q = 0; q < CACHE_DEPTH*16*BLOCK_SIZE+(1-CHANNELS_FIRST)*CACHE_DEPTH; ++q)
                            LDS_[X_OFFSET + q] = -1.0;
                    }
                    #endif
                }

                GroupMemoryBarrierWithGroupSync();

                int4 idX = int4(0,1,2,3);
                int4 idW = int4(0,16,32,48);
                int incX = 64 + (1-CHANNELS_FIRST);
                int incW = 64;

                for (int di = 0; di < CACHE_DEPTH; di++)
                {
                    float4 srcX = float4(
                        LDS_[X_OFFSET + idX.x + ty*4],
                        LDS_[X_OFFSET + idX.y + ty*4],
                        LDS_[X_OFFSET + idX.z + ty*4],
                        LDS_[X_OFFSET + idX.w + ty*4]);
                    float4 srcW = float4(
                        LDS_[W_OFFSET + idW.x + tx],
                        LDS_[W_OFFSET + idW.y + tx],
                        LDS_[W_OFFSET + idW.z + tx],
                        LDS_[W_OFFSET + idW.w + tx]
                    );
                    idX += incX;
                    idW += incW;

                    dstA[0].x = ffma(srcX.x, srcW.x, dstA[0].x);
                    dstA[0].y = ffma(srcX.x, srcW.y, dstA[0].y);
                    dstA[0].z = ffma(srcX.x, srcW.z, dstA[0].z);
                    dstA[0].w = ffma(srcX.x, srcW.w, dstA[0].w);

                    dstA[1].x = ffma(srcX.y, srcW.x, dstA[1].x);
                    dstA[1].y = ffma(srcX.y, srcW.y, dstA[1].y);
                    dstA[1].z = ffma(srcX.y, srcW.z, dstA[1].z);
                    dstA[1].w = ffma(srcX.y, srcW.w, dstA[1].w);

                    dstA[2].x = ffma(srcX.z, srcW.x, dstA[2].x);
                    dstA[2].y = ffma(srcX.z, srcW.y, dstA[2].y);
                    dstA[2].z = ffma(srcX.z, srcW.z, dstA[2].z);
                    dstA[2].w = ffma(srcX.z, srcW.w, dstA[2].w);

                    dstA[3].x = ffma(srcX.w, srcW.x, dstA[3].x);
                    dstA[3].y = ffma(srcX.w, srcW.y, dstA[3].y);
                    dstA[3].z = ffma(srcX.w, srcW.z, dstA[3].z);
                    dstA[3].w = ffma(srcX.w, srcW.w, dstA[3].w);
                }

                GroupMemoryBarrierWithGroupSync();
            }
        }
    }

    #if SHUFFLE_FOR_COALESCED_STORE && CHANNELS_FIRST && STRICT_CHANNELS
        //-----------------------------------------------------
        //Use LDS to shuffle TG registers into coalesced writes
        //-----------------------------------------------------
        //A TG output [64pixels,64channels] = 4096 values. We will process [32,64] values at a time per TG.
        for (uint tg_registerChannelOffset = 0; tg_registerChannelOffset < BLOCK_SIZE; tg_registerChannelOffset += 2)
        {
            uint tg_kId;
            uint tg_pId;
            //Store 4 pixels x 2 channels per threads to LDS.
            uint ldsRowOffset = groupThreadID.x * 2;
            uint ldsPixelOffset = groupThreadID.y * BLOCK_SIZE;
            [unroll] for (tg_kId = 0; tg_kId < 2; ++tg_kId)
                [unroll] for (tg_pId = 0; tg_pId < BLOCK_SIZE; ++tg_pId)
                {
                    LDS_[64 * (groupThreadID.x * 2 + tg_kId) + ldsPixelOffset + tg_pId] = dstA[tg_pId][tg_registerChannelOffset + tg_kId];
                }

            GroupMemoryBarrierWithGroupSync();

            //We have a buffers of [64pixels,32channels] floats, each thread will store [1pixels,8channels] so a threadgroup is storing 64 pixels and 4 channels at a time to DDR in a linear fashion.
            uint readPixelId = threadIndex % 64;
            uint writePixelId = outputPixelBaseId + readPixelId;

            if (writePixelId < numOuputPixels)
            {
                [unroll] for (tg_kId = 0; tg_kId < 32; tg_kId+=4)
                {
                    uint readChannelId = tg_kId + threadIndex / 64;
                    uint readIndex = 64 * readChannelId + readPixelId;
                    uint writeChannelId = kernelBaseId + readChannelId%2 + (readChannelId/2)*BLOCK_SIZE + tg_registerChannelOffset;
                    O.FastSetWithActivation(h*w* writeChannelId + writePixelId + offsetO + batchWriteOffset, LDS_[readIndex]);
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }
    #else
        #if CHANNELS_FIRST
            [unroll] for (int sy = 0; sy < 4 && y+sy < (int)w * (int)h; ++sy)
                [unroll] for (int sx = 0; sx < 4 && x+sx < strideO; ++sx)
                    O.FastSetWithActivation(h*w* (x+sx) + (y+sy) + offsetO + batchWriteOffset, dstA[sy][sx]);
        #else
        [unroll] for (int sy = 0; sy < 4 && y+sy < (int)w * (int)h * (int)O.batch; ++sy)
            [unroll] for (int sx = 0; sx < 4 && x+sx < strideO; ++sx)
                O.FastSetWithActivation(strideO * (y+sy) + x+sx + offsetO, dstA[sy][sx]);
        #endif
    #endif


    #undef X_
    #undef W_
    #undef LDS_
    #undef X_OFFSET
    #undef W_OFFSET
}
#undef CACHE_DEPTH
#undef BUF_OFFSET
#endif
#undef KERNEL_NAME

NUMTHREADS((16,4,4), (8,4,4), (4,4,4))
void KERNEL_FUNC(Conv2D)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(K.kernelCount, O.width, O.height);
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);

    uint k = dispatchThreadID.x;
    uint x = dispatchThreadID.y;
    uint y = dispatchThreadID.z;

    if (k >= K.channels) return;
    if (x >= O.width) return;
    if (y >= O.height) return;

    uint2 leftCorner = _Pad.xy;
    uint2 rightCorner = uint2(X.width, X.height) + _Pad.xy;
    for (uint n = 0; n < O.batch; ++n)
    {
        float acc = B.FastGet(k);
        for (uint dy = 0; dy < K.GetKernelHeight(); ++dy)
        {
            for (uint dx = 0; dx < K.GetKernelWidth(); ++dx)
            {
                uint2 pos = uint2(x, y) * _Stride.xy + uint2(dx, dy);

                for (uint c = 0; c < X.channels; ++c)
                {
                    float v = 0;

                    // WARNING: Mali-G71 performance drops 4x if this branching includes storing accumulator
                    if (!any(pos < leftCorner) && !any(pos >= rightCorner))
                        v = X.Get(n, pos.y - leftCorner.y, pos.x - leftCorner.x, c);
                    //acc = fastfma(v,  K.Get(dy, dx, c, k), acc);
                    acc += v * K.Get(dy, dx, c, k);
                }
            }
        }

        O.SetWithActivation(n, y, x, k, acc);
    }
}


#define SIZE_W 4
#define SIZE_H 2
NUMTHREADS((64, 2, 2), (32, 2, 2), (16, 2, 2))
void KERNEL_FUNC(Conv2D_RegisterBlock4x2)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(K.kernelCount, O.width, O.height);
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);

    uint k = dispatchThreadID.x;
    uint x = dispatchThreadID.y;
    uint y = dispatchThreadID.z;

    if (k >= K.channels) return;
    if (x*SIZE_W >= O.width) return;
    if (y*SIZE_H >= O.height) return;

    uint2 leftCorner = _Pad.xy;
    uint2 rightCorner = uint2(X.width, X.height) + _Pad.xy;
    for (uint n = 0; n < O.batch; ++n)
    {
        float acc[SIZE_H*SIZE_W];
        uint q;
        [unroll]
        for (q = 0; q < SIZE_H*SIZE_W; ++q)
            acc[q] = B.FastGet(k);
        for (uint dy = 0; dy < K.GetKernelHeight(); ++dy)
        {
            for (uint dx = 0; dx < K.GetKernelWidth(); ++dx)
            {
                uint2 pos[SIZE_H*SIZE_W];
                [unroll]
                for (q = 0; q < SIZE_H*SIZE_W; ++q)
                    pos[q] = uint2(x*SIZE_W+(q%SIZE_W), y*SIZE_H+(q/SIZE_W)) * _Stride.xy + uint2(dx, dy);

                for (uint c = 0; c < X.channels; ++c)
                    [unroll]
                    for (q = 0; q < SIZE_H*SIZE_W; ++q)
                        if (all(pos[q] >= leftCorner) && all(pos[q] < rightCorner))
                            acc[q] = fastfma(X.Get(n, pos[q] - leftCorner, c), K.Get(dy, dx, c, k), acc[q]);
            }
        }

        [unroll]
        for (q = 0; q < SIZE_H*SIZE_W; ++q)
            O.SetWithActivation(n, y*SIZE_H+(q/SIZE_W), x*SIZE_W+(q%SIZE_W), k, acc[q]);
    }
}
#undef SIZE_W
#undef SIZE_H

//DISPATCH ARGS(K.kernelCount, O.width, O.height);
#define CONV2D_L1CACHED(L1CACHESIZE, SIZE, FMA) \
groupshared float Conv2D_L1Cached##L1CACHESIZE##_Reg_Loop_safe_X[SIZE*SIZE][L1CACHESIZE];\
[numthreads(L1CACHESIZE, 1, 1)]\
void KERNEL_FUNC(Conv2D_L1Cached##L1CACHESIZE##_RegisterBlock##SIZE##x##SIZE)(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)\
{\
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);\
\
    uint k = L1CACHESIZE * groupID.x + groupThreadID.x;\
    uint x = groupID.y;\
    uint y = groupID.z;\
\
    if (x*SIZE >= O.width) return;\
    if (y*SIZE >= O.height) return;\
\
    for (uint n = 0; n < O.batch; ++n)\
    {\
        float acc[SIZE*SIZE];\
        uint q;\
        [unroll]\
        for (q = 0; q < SIZE*SIZE; ++q)\
            acc[q] = B.SafeGet(k);\
\
        for (uint dy = 0; dy < K.GetKernelHeight(); ++dy)\
        {\
            for (uint dx = 0; dx < K.GetKernelWidth(); ++dx)\
            {\
                uint2 pos[SIZE*SIZE];\
                [unroll]\
                for (q = 0; q < SIZE*SIZE; ++q)\
                    pos[q] = uint2(x*SIZE+(q%SIZE), y*SIZE+(q/SIZE)) * _Stride.xy + uint2(dx, dy);\
\
                for (uint c = 0; c < X.channels; c += L1CACHESIZE)\
                {\
                    uint dc = groupThreadID.x;\
                    [unroll]\
                    for (q = 0; q < SIZE*SIZE; ++q)\
                        Conv2D_L1Cached##L1CACHESIZE##_Reg_Loop_safe_X[q][dc] = X.SafeGet(n, pos[q], c + dc, _Pad.xy);\
                    GroupMemoryBarrierWithGroupSync();\
\
                    if (k < K.channels)\
                    {\
                        uint kIndex = K.IndexHWC(dy, dx, c, k);\
                        for (dc = 0; dc < L1CACHESIZE; ++dc)\
                        {\
                            [unroll]\
                            for (q = 0; q < SIZE*SIZE; ++q)\
                                acc[q] = FMA(Conv2D_L1Cached##L1CACHESIZE##_Reg_Loop_safe_X[q][dc], K.data[kIndex], acc[q]);\
                            kIndex += K.channels;\
                        }\
                    }\
                    GroupMemoryBarrierWithGroupSync();\
                }\
            }\
        }\
\
        uint remainderW = (O.width - x*SIZE);\
        uint remainderH = (O.height - y*SIZE);\
\
        if (k < K.channels)\
            [unroll]\
            for (q = 0; q < SIZE*SIZE; ++q)\
                if (q/SIZE < remainderH && q%SIZE < remainderW)\
                    O.SetWithActivation(n, y*SIZE+(q/SIZE), x*SIZE+(q%SIZE), k, acc[q]);\
    }\
\
}

CONV2D_L1CACHED(64,4, fastfma)
CONV2D_L1CACHED(32,4, fastfma)


// IDEA: iterate over channels in the inner loop - needs channels first layout
NUMTHREADS((16,4,4), (8,4,4), (4,4,4))
void KERNEL_FUNC(DepthwiseConv2D)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(K.kernelCount, O.width, O.height);
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);

    uint k = dispatchThreadID.x;
    uint x = dispatchThreadID.y;
    uint y = dispatchThreadID.z;

    if (k >= K.channels) return;
    if (x >= O.width) return;
    if (y >= O.height) return;

    uint2 leftCorner = _Pad.xy;
    uint2 rightCorner = uint2(X.width, X.height) + _Pad.xy;

    uint2 leftKernelCorner = uint2(x, y) * _Stride.xy;
    uint2 rightKernelCorner = leftKernelCorner + uint2(K.GetKernelWidth(), K.GetKernelHeight());

    if (any(leftKernelCorner < leftCorner) || any(rightKernelCorner >= rightCorner))
    {
        // path with edge-cases checks
        for (uint n = 0; n < O.batch; ++n)
        {
            float acc = B.FastGet(k);
            for (uint dy = 0; dy < K.GetKernelHeight(); ++dy)
                for (uint dx = 0; dx < K.GetKernelWidth(); ++dx)
                {
                    uint2 pos = leftKernelCorner + uint2(dx, dy);
                    if (any(pos < leftCorner)) continue;
                    if (any(pos >= rightCorner)) continue;

                    acc = fastfma(
                        X.Get(n, pos.y - leftCorner.y, pos.x - leftCorner.x, k),
                        K.Get(dy, dx, 0, k),
                        acc);
                }

            O.SetWithActivation(n, y, x, k, acc);
        }
    }
    else
    {
        // kernel is guaranteed to be within X,
        // no need to check against edge-cases
        leftKernelCorner -= leftCorner;
        for (uint n = 0; n < O.batch; ++n)
        {
            float acc = B.FastGet(k);
            for (uint dy = 0; dy < K.GetKernelHeight(); ++dy)
                for (uint dx = 0; dx < K.GetKernelWidth(); ++dx)
                {
                    uint2 pos = leftKernelCorner + uint2(dx, dy);

                    acc = fastfma(
                        X.Get(n, pos, k),
                        K.Get(dy, dx, 0, k),
                        acc);
                }

            O.SetWithActivation(n, y, x, k, acc);
        }
    }
}


NUMTHREADS((16, 4, 4), (8, 4, 4), (4, 4, 4))
void Conv2DTransFlipKernel(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_SHARED_MODEL(K, WBK); TENSOR_SHARED_MODEL(B, WBK); TENSOR_ARG_RW(O)

    uint k = dispatchThreadID.x;
    uint c = dispatchThreadID.y;
    uint z = dispatchThreadID.z; // x + KWidth * y

    uint x = z % K.GetKernelWidth();
    uint y = z / K.GetKernelWidth();

    if (c >= K.GetKernelDepth()) return;
    if (k >= K.GetKernelCount()) return;
    if (z >= K.GetKernelHeight() * K.GetKernelWidth()) return;

    float v = K.Get(K.GetKernelHeight() - 1 - y, K.GetKernelWidth() - 1 - x, c, k);
    O.Set(y, x, c, k, v);
    O.FastSet(K.GetLength() + k, B.FastGet(k));
}

NUMTHREADS((16, 4, 4), (8, 4, 4), (4, 4, 4))
void KERNEL_FUNC(Conv2DTransPadFill)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(X.channels, X.width, X.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint x = dispatchThreadID.y;
    uint y = dispatchThreadID.z;

    if (c >= X.channels) return;
    if (x >= X.width) return;
    if (y >= X.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        uint ox = x * _Stride.x;
        uint oy = y * _Stride.y;

        uint strideX = x == (X.width - 1)  ? _Pad.x + 1 : _Stride.x;
        uint strideY = y == (X.height - 1) ? _Pad.y + 1 : _Stride.y;

        for (uint dx = 0; dx < strideX; dx++)
            for (uint dy = 0; dy < strideY; dy++)
            {
                O.Set(n, oy + dy, ox + dx, c, 0.0f);
            }
        float v = X.Get(n, y, x, c);
        O.Set(n, oy, ox, c, v);
    }
}

[numthreads(4,4,4)]
void KERNEL_FUNC(Conv2DTrans)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(K.kernelCount, O.width, O.height);
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);

    uint k = dispatchThreadID.x;
    uint x = dispatchThreadID.y;
    uint y = dispatchThreadID.z;

    if (k >= K.channels) return;
    if (x >= O.width) return;
    if (y >= O.height) return;

    uint2 strideMask = _Stride.xy - 1;

    for (uint n = 0; n < O.batch; ++n)
    {
        float acc = B.FastGet(k);
        for (uint dy = (y + _Pad.y) & strideMask.y; dy < K.GetKernelHeight(); dy += _Stride.y)
        {
            for (uint dx = (x + _Pad.x) & strideMask.x; dx < K.GetKernelWidth(); dx += _Stride.x)
            {

                uint2 pos = uint2(x + dx, y + dy);
                uint2 opos = (pos - _Pad.xy) / _Stride.xy;

                if (any(opos >= uint2(X.width, X.height))) continue;
                if (any(pos < _Pad.xy)) continue;

                for (uint c = 0; c < X.channels; ++c)
                {
                    acc = fastfma(  X.Get(n, opos.y, opos.x, c),
                                    K.Get(  K.GetKernelHeight() - 1 - dy,
                                            K.GetKernelWidth()  - 1 - dx, c, k),
                                    acc);
                }
            }
        }

        O.SetWithActivation(n, y, x, k, acc);
    }
}

#if defined(MAX_KERNEL_SIZE) && defined(GROUP_SIZE_X) && defined(GROUP_SIZE_Y)

#if CHANNELS_FIRST
    #define CONV2DTRANS_NAME_CALL(KERNEL,TGX,TGY) Conv2DTrans_KernelCached_K##KERNEL##x##KERNEL##_T##TGX##x##TGY##_NCHW
#else
    #define CONV2DTRANS_NAME_CALL(KERNEL,TGX,TGY) Conv2DTrans_KernelCached_K##KERNEL##x##KERNEL##_T##TGX##x##TGY##_NHWC
#endif
#define CONV2DTRANS_NAME(KERNEL,TGX,TGY) CONV2DTRANS_NAME_CALL(KERNEL,TGX,TGY)
groupshared float Conv2DTrans_SharedKernel[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][GROUP_SIZE_X*GROUP_SIZE_Y];
groupshared float Conv2DTrans_SharedBias;
[numthreads(1,GROUP_SIZE_X,GROUP_SIZE_Y)]
void CONV2DTRANS_NAME(MAX_KERNEL_SIZE, GROUP_SIZE_X,GROUP_SIZE_Y)(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex: SV_GroupIndex)
{
    //Constraints:
    // C <= GROUP_SIZE_X*GROUP_SIZE_Y
    // K <= MAX_KERNEL_SIZExMAX_KERNEL_SIZE
    //DISPATCH ARGS(K.kernelCount, O.width, O.height);
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);

    uint k = dispatchThreadID.x;
    uint x = dispatchThreadID.y;
    uint y = dispatchThreadID.z;

    //Dispatch organisation:
    //  a thread  = write to [:,y,x,k] ie all batch but a single 2d pos and feature.
    //  a thread group = handle 1 feature in a GROUP_SIZExGROUP_SIZE x,y region, it loop other all batch, input channel count need to be <= GROUP_SIZE*GROUP_SIZE

    //LDS allocation
    //  we have 1 feature and up to GROUP_SIZE_X*GROUP_SIZE_Y channels per thread group, batch all use the same kernels,
    //  thus LDS is [MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][GROUP_SIZE_X*GROUP_SIZE_Y]

    //Loading to LDS
    //  Each threads load a 2D kernel for a different channel into LDS
    for(uint dy = 0; dy < K.GetKernelWidth(); ++dy)
    {
        for(uint dx = 0; dx < K.GetKernelHeight(); ++dx)
        {
            uint channelToLoadIndex = groupIndex;
            if((channelToLoadIndex < X.channels) && (k < K.channels))
                Conv2DTrans_SharedKernel[dy][dx][channelToLoadIndex] = K.Get(K.GetKernelHeight() - 1 - dy,K.GetKernelWidth() - 1 - dx, channelToLoadIndex, k);
        }
    }
    //  first thread also load bias to LDS
    if (groupIndex == 0)
        Conv2DTrans_SharedBias = B.FastGet(k);

    //Wait for all load to complete
    GroupMemoryBarrierWithGroupSync();

    // Outside of target tensor, nothing to write to or compute exit.
    if (x >= O.width) return;
    if (y >= O.height) return;
    if (k >= K.channels) return;

    // Apply kernels from LDS to all batches and write result out (per batch as input differ)
    uint2 strideMask = _Stride.xy - 1;
    for (uint n = 0; n < O.batch; ++n)
    {
        float acc = Conv2DTrans_SharedBias;
        for (uint dy = (y + _Pad.y) & strideMask.y; dy < K.GetKernelHeight(); dy += _Stride.y)
        {
            for (uint dx = (x + _Pad.x) & strideMask.x; dx < K.GetKernelWidth(); dx += _Stride.x)
            {
                uint2 pos = uint2(x + dx, y + dy);
                uint2 opos = (pos - _Pad.xy) / _Stride.xy;
                if (any(opos >= uint2(X.width, X.height))) continue;
                if (any(pos < _Pad.xy)) continue;

                for (uint c = 0; c < X.channels; ++c)
                {
                    acc = fastfma(X.Get(n, opos.y, opos.x, c),
                        Conv2DTrans_SharedKernel[dy][dx][c],
                        acc);
                }
            }
        }
        O.SetWithActivation(n, y, x, k, acc);
    }
}
#undef CONV2DTRANS_NAME
#endif //defined(MAX_KERNEL_SIZE) && defined(GROUP_SIZE_X) && defined(GROUP_SIZE_Y)




// https://github.com/andravin/wincnn
// https://arxiv.org/pdf/1509.09308.pdf
// Winograd: 4x4 image, 3x3 kernel, 2x2 output
static const float4x3 Winograd_G = float4x3(float3(1, 0, 0), float3(0.5, 0.5, 0.5), float3(0.5, -0.5, 0.5), float3(0, 0, 1));
static const float3x4 Winograd_GT = transpose(Winograd_G);

NUMTHREADS((16, 4, 4), (8, 4, 4), (4, 4, 4))
void KernelWinograd_3x3(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_SHARED_MODEL(K, WBK); TENSOR_SHARED_MODEL(B, WBK); TENSOR_ARG_RW(O)

    uint k = dispatchThreadID.x;
    uint c = dispatchThreadID.y;
    uint i = dispatchThreadID.z;

    if (c >= K.GetKernelDepth()) return;
    if (k >= K.GetKernelCount()) return;

    float3x3 g;
    g[0][0] = K.Get(0, 0, c, k);
    g[0][1] = K.Get(0, 1, c, k);
    g[0][2] = K.Get(0, 2, c, k);
    g[1][0] = K.Get(1, 0, c, k);
    g[1][1] = K.Get(1, 1, c, k);
    g[1][2] = K.Get(1, 2, c, k);
    g[2][0] = K.Get(2, 0, c, k);
    g[2][1] = K.Get(2, 1, c, k);
    g[2][2] = K.Get(2, 2, c, k);

    float4x4 v = mul(Winograd_G, mul(g, Winograd_GT));

    O.Set(0, 0, c, k, v[0][0]);
    O.Set(1, 0, c, k, v[1][0]);
    O.Set(2, 0, c, k, v[2][0]);
    O.Set(3, 0, c, k, v[3][0]);
    O.Set(0, 1, c, k, v[0][1]);
    O.Set(1, 1, c, k, v[1][1]);
    O.Set(2, 1, c, k, v[2][1]);
    O.Set(3, 1, c, k, v[3][1]);
    O.Set(0, 2, c, k, v[0][2]);
    O.Set(1, 2, c, k, v[1][2]);
    O.Set(2, 2, c, k, v[2][2]);
    O.Set(3, 2, c, k, v[3][2]);
    O.Set(0, 3, c, k, v[0][3]);
    O.Set(1, 3, c, k, v[1][3]);
    O.Set(2, 3, c, k, v[2][3]);
    O.Set(3, 3, c, k, v[3][3]);

    uint kLength = (K.GetKernelHeight() + 1) * (K.GetKernelWidth() + 1) * K.GetKernelDepth() * K.GetKernelCount();
    if (i < B.GetLength())
        O.FastSet(kLength + i, B.FastGet(i));
}

float4x4 ApplyWinnogradB(float4x4 d)
{
    // BT x u x B, used mathematica to express the operation using only +/-
    //return float4x4(float4( d[0][0] - d[0][2] - d[2][0] + d[2][2],  d[0][1] + d[0][2] - d[2][1] - d[2][2], -d[0][1] + d[0][2] + d[2][1] - d[2][2], -d[0][1] + d[0][3] + d[2][1] - d[2][3]),
    //                float4( d[1][0] - d[1][2] + d[2][0] - d[2][2],  d[1][1] + d[1][2] + d[2][1] + d[2][2], -d[1][1] + d[1][2] - d[2][1] + d[2][2], -d[1][1] + d[1][3] - d[2][1] + d[2][3]),
    //                float4(-d[1][0] + d[1][2] + d[2][0] - d[2][2], -d[1][1] - d[1][2] + d[2][1] + d[2][2],  d[1][1] - d[1][2] - d[2][1] + d[2][2],  d[1][1] - d[1][3] - d[2][1] + d[2][3]),
    //                float4(-d[1][0] + d[1][2] + d[3][0] - d[3][2], -d[1][1] - d[1][2] + d[3][1] + d[3][2],  d[1][1] - d[1][2] - d[3][1] + d[3][2],  d[1][1] - d[1][3] - d[3][1] + d[3][3])
    //    );
    // re-order operations to lower register pressure
    float4x4 TU;
    float4x4 U;
    TU[0][0] = d[0][0] - d[2][0];
    TU[0][1] = d[0][1] - d[2][1];
    TU[0][2] = d[0][2] - d[2][2];
    TU[0][3] = d[0][3] - d[2][3];

    TU[1][0] = d[1][0] + d[2][0];
    TU[1][1] = d[1][1] + d[2][1];
    TU[1][2] = d[1][2] + d[2][2];
    TU[1][3] = d[1][3] + d[2][3];

    TU[2][0] = d[2][0] - d[1][0];
    TU[2][1] = d[2][1] - d[1][1];
    TU[2][2] = d[2][2] - d[1][2];
    TU[2][3] = d[2][3] - d[1][3];

    TU[3][0] = d[3][0] - d[1][0];
    TU[3][1] = d[3][1] - d[1][1];
    TU[3][2] = d[3][2] - d[1][2];
    TU[3][3] = d[3][3] - d[1][3];


    U[0][0] = TU[0][0] - TU[0][2];
    U[0][1] = TU[0][1] + TU[0][2];
    U[0][2] = TU[0][2] - TU[0][1];
    U[0][3] = TU[0][3] - TU[0][1];

    U[1][0] = TU[1][0] - TU[1][2];
    U[1][1] = TU[1][1] + TU[1][2];
    U[1][2] = TU[1][2] - TU[1][1];
    U[1][3] = TU[1][3] - TU[1][1];

    U[2][0] = TU[2][0] - TU[2][2];
    U[2][1] = TU[2][1] + TU[2][2];
    U[2][2] = TU[2][2] - TU[2][1];
    U[2][3] = TU[2][3] - TU[2][1];

    U[3][0] = TU[3][0] - TU[3][2];
    U[3][1] = TU[3][1] + TU[3][2];
    U[3][2] = TU[3][2] - TU[3][1];
    U[3][3] = TU[3][3] - TU[3][1];

    return U;
}

float2x2 ApplyWinnogradA(float4x4 uv)
{
    // A x u x A, used mathematica to express the operation using only +/-
    // return float2x2(float2(uv[0][0] + uv[0][1] + uv[0][2] + uv[1][0] + uv[1][1] + uv[1][2] + uv[2][0] + uv[2][1] + uv[2][2], uv[0][1] - uv[0][2] + uv[0][3] + uv[1][1] - uv[1][2] + uv[1][3] + uv[2][1] - uv[2][2] + uv[2][3]),
    //                 float2(uv[1][0] + uv[1][1] + uv[1][2] - uv[2][0] - uv[2][1] - uv[2][2] + uv[3][0] + uv[3][1] + uv[3][2], uv[1][1] - uv[1][2] + uv[1][3] - uv[2][1] + uv[2][2] - uv[2][3] + uv[3][1] - uv[3][2] + uv[3][3])
    //                );
    // re-order operations to lower register pressure
    float2x4 TY;
    float2x2 Y;
    TY[0][0] = uv[0][0] + uv[0][1] + uv[0][2];
    TY[0][1] = uv[1][0] + uv[1][1] + uv[1][2];
    TY[0][2] = uv[2][0] + uv[2][1] + uv[2][2];
    TY[0][3] = uv[3][0] + uv[3][1] + uv[3][2];

    TY[1][0] = uv[0][1] - uv[0][2] + uv[0][3];
    TY[1][1] = uv[1][1] - uv[1][2] + uv[1][3];
    TY[1][2] = uv[2][1] - uv[2][2] + uv[2][3];
    TY[1][3] = uv[3][1] - uv[3][2] + uv[3][3];


    Y[0][0] = TY[0][0] + TY[0][1] + TY[0][2];
    Y[0][1] = TY[1][0] + TY[1][1] + TY[1][2];
    Y[1][0] = TY[0][1] - TY[0][2] + TY[0][3];
    Y[1][1] = TY[1][1] - TY[1][2] + TY[1][3];

    return Y;
}

#undef KERNEL_NAME
#undef FUNC_NAME_CALL
#undef CACHE_NAME_CALL
#undef FUNC_NAME
#undef CACHE_NAME

#define KERNEL_NAME Conv2DWinograd_2x2_
#if CHANNELS_FIRST
    #define FUNC_NAME_CALL(KERNEL, SUFFIX, SIZE_K, SIZE_X) KERNEL##SUFFIX##SIZE_K##x##SIZE_X##_NCHW
    #define CACHE_NAME_CALL(KERNEL, SUFFIX, SIZE_K, SIZE_X, TENSOR) KERNEL##SUFFIX##SIZE_K##x##SIZE_X##_Cache_##TENSOR##_NCHW
#else
    #define FUNC_NAME_CALL(KERNEL, SUFFIX, SIZE_K, SIZE_X) KERNEL##SUFFIX##SIZE_K##x##SIZE_X##_NHWC
    #define CACHE_NAME_CALL(KERNEL, SUFFIX, SIZE_K, SIZE_X, TENSOR) KERNEL##SUFFIX##SIZE_K##x##SIZE_X##_Cache_##TENSOR##_NHWC
#endif
#define FUNC_NAME(KERNEL, SUFFIX, SIZE_K, SIZE_X) FUNC_NAME_CALL(KERNEL, SUFFIX, SIZE_K, SIZE_X)
#define CACHE_NAME(KERNEL, SUFFIX, SIZE_K, SIZE_X, TENSOR) CACHE_NAME_CALL(KERNEL, SUFFIX, SIZE_K, SIZE_X, TENSOR)

#if BLOCK_SIZE == 4
#if KERNEL_PER_TG == 16
//NCHW
#define CACHE_DEPTH 8

#define CACHE_WIDTH_X 16
#define CACHE_WIDTH_W 16


groupshared float CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, BLOCK_SIZE, LDS)[4576];


[numthreads(256, 1, 1)]
void FUNC_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, BLOCK_SIZE)(uint3 groupID : SV_GroupID, uint threadIndexGlobal : SV_GroupIndex)
{
    //This kernel assume the following:
    //Input:
    //Ouput:
    //Kernel:
    //DISPATCH ARGS(K.kernelCount, O.width * O.height, O.batch);
    TENSOR_SHARED2_ARGS4(X, K, B, WBK, O);
#define LDS_ CACHE_NAME(KERNEL_NAME, SUFFIX, BLOCK_SIZE, BLOCK_SIZE, LDS)
#define X_OFFSET 0
#define W_OFFSET 16*CACHE_DEPTH*CACHE_WIDTH_X

    //Per thread group (scalar registers)
    uint tg_NumChannels = X.channels;
    uint tg_WidthX = X.width;
    uint tg_HeightX = X.height;
    uint tg_WidthO = O.width;
    uint tg_HeightO = O.height;
    uint tg_WidthOHalf = (tg_WidthO + 1) / 2;
    uint tg_NumKernels = K.channels;
    uint tg_NumInputPixels = tg_WidthX * tg_HeightX;
    uint tg_NumOuputPixels = tg_WidthO * tg_HeightO;
    uint tg_KernelSpatialStride = tg_NumKernels * tg_NumChannels;
    uint tg_KernelBaseId = groupID.x * CACHE_WIDTH_W;
    uint tg_OutputPixelBaseId = groupID.y * CACHE_WIDTH_X;
    uint tg_BatchReadOffset = groupID.z * tg_NumChannels * tg_HeightX * tg_WidthX;
    uint tg_BatchWriteOffset = groupID.z * tg_NumKernels * tg_HeightO * tg_WidthO;

    // output per TG: 4 pixels x 4 features x 4x4 threads x (2x2 pixel blocks) => 64 pixels x 16 features
    // LDS is 256 * 4x4 in order to hold 256 (8 * 4 * 4 x 2) patches of 4x4 for inverse winograd transform of X and W
    // 16 (4x4 parallel matmuls) * 8 (cache_depth) * 2 (K and X) * 16 (4x4 block)

    // threadIndex4x4 = threadIndexGlobal/16 : 16 SGEM (4x4 patch of X) in parallel
    // threadIndex = threadIndexGlobal%16 : 4x4 threads for one SGEM, this is divided into pixels and features (groupThreadIDY = threadIndex/4, groupThreadIDX = threadIndex%4)
    uint threadIndex4x4 = (threadIndexGlobal >> 4);
    uint threadIndex = (threadIndexGlobal & 0xf);
    uint groupThreadIDY4 = (threadIndexGlobal & 0xc); // groupThreadIDY * 4
    uint groupThreadIDX4 = ((threadIndexGlobal & 0x3) << 2); // groupThreadIDX * 4

    // 4x4 block, 4 kernels by 4 pixels
    //**********************************
    //* Kernel Ids  *  0  1  2  3  ...
    //**********************************
    //              *  ThreadIds
    // Pixel Ids  0 *  0  1  2  3
    //            1 *  8  9 10 11
    //            2 * 16 17 18 19
    //            3 * 32 33 34 35
    float dstA[BLOCK_SIZE*BLOCK_SIZE];

    // Load Bias [K] int dstA [Kernels, Pixels]
    dstA[0*BLOCK_SIZE + 0] = 0;
    dstA[0*BLOCK_SIZE + 1] = 0;
    dstA[0*BLOCK_SIZE + 2] = 0;
    dstA[0*BLOCK_SIZE + 3] = 0;
    dstA[1*BLOCK_SIZE + 0] = 0;
    dstA[1*BLOCK_SIZE + 1] = 0;
    dstA[1*BLOCK_SIZE + 2] = 0;
    dstA[1*BLOCK_SIZE + 3] = 0;
    dstA[2*BLOCK_SIZE + 0] = 0;
    dstA[2*BLOCK_SIZE + 1] = 0;
    dstA[2*BLOCK_SIZE + 2] = 0;
    dstA[2*BLOCK_SIZE + 3] = 0;
    dstA[3*BLOCK_SIZE + 0] = 0;
    dstA[3*BLOCK_SIZE + 1] = 0;
    dstA[3*BLOCK_SIZE + 2] = 0;
    dstA[3*BLOCK_SIZE + 3] = 0;


    for (uint tg_ChannelOffset = 0; tg_ChannelOffset < tg_NumChannels; tg_ChannelOffset += CACHE_DEPTH)
    {
        // Load from DDR to LDS: 1 SGEMM : (4*4 weight + 4*4 pixel) * CACHE_DEPTH => 1024 Bytes
        // => x16 SGEMM = 16384 Bytes
        // Storing in registers to avoid sync inside the loop.
        // LOAD W and X in registers and perform Winograd transform
        if (threadIndexGlobal < 128) // threadIndex4x4 < 8
        {
            uint threadIndexHigh = threadIndex4x4;

            float4x4 tempX;
            uint tg_Dy;
            uint tg_Dx;
            [unroll] for (tg_Dy = 0; tg_Dy < BLOCK_SIZE; tg_Dy++)
            {
                [unroll] for (tg_Dx = 0; tg_Dx < BLOCK_SIZE; tg_Dx++)
                {
                    uint outputPixelBaseId = tg_OutputPixelBaseId + threadIndex;
                    uint2 outputPixelCoords = 2 * uint2(outputPixelBaseId % tg_WidthOHalf, outputPixelBaseId / tg_WidthOHalf);

                    uint2 inputPixelCoords = outputPixelCoords - _Pad.xy + uint2(tg_Dx, tg_Dy);

                    bool inputPixelMask = all(inputPixelCoords < uint2(tg_WidthX, tg_HeightX));

                    int inputPixelId = inputPixelCoords.y * tg_WidthX + inputPixelCoords.x;
                    uint inputChannelId = tg_ChannelOffset + threadIndexHigh;

                    uint pixelReadOffset = tg_NumInputPixels * inputChannelId + inputPixelId + tg_BatchReadOffset;

                    tempX[tg_Dy][tg_Dx] = X.MaskedGet(inputPixelMask, pixelReadOffset);
                }
            }
            tempX = ApplyWinnogradB(tempX);

            // store tempX interleaved per thread:
            // thread: 0 1 2 .... 128 0 1 2 .... 128 [16SGEMM x (8 values)]
            //         <- tempX[0] -> <- tempX[1] ->
            // to avoid bank conflict in the inner loop, we shift every tempX by 18*8 instead of 256=16*8
            // LDS_[([0,15])*18*8 + (threadIndexGlobal/16)*16 + (threadIndexGlobal%16)] = tempX[[0,15]]
            LDS_[((0 * 4 + 0) * 18 * 8) + threadIndexGlobal] = tempX[0][0];
            LDS_[((0 * 4 + 1) * 18 * 8) + threadIndexGlobal] = tempX[0][1];
            LDS_[((0 * 4 + 2) * 18 * 8) + threadIndexGlobal] = tempX[0][2];
            LDS_[((0 * 4 + 3) * 18 * 8) + threadIndexGlobal] = tempX[0][3];
            LDS_[((1 * 4 + 0) * 18 * 8) + threadIndexGlobal] = tempX[1][0];
            LDS_[((1 * 4 + 1) * 18 * 8) + threadIndexGlobal] = tempX[1][1];
            LDS_[((1 * 4 + 2) * 18 * 8) + threadIndexGlobal] = tempX[1][2];
            LDS_[((1 * 4 + 3) * 18 * 8) + threadIndexGlobal] = tempX[1][3];
            LDS_[((2 * 4 + 0) * 18 * 8) + threadIndexGlobal] = tempX[2][0];
            LDS_[((2 * 4 + 1) * 18 * 8) + threadIndexGlobal] = tempX[2][1];
            LDS_[((2 * 4 + 2) * 18 * 8) + threadIndexGlobal] = tempX[2][2];
            LDS_[((2 * 4 + 3) * 18 * 8) + threadIndexGlobal] = tempX[2][3];
            LDS_[((3 * 4 + 0) * 18 * 8) + threadIndexGlobal] = tempX[3][0];
            LDS_[((3 * 4 + 1) * 18 * 8) + threadIndexGlobal] = tempX[3][1];
            LDS_[((3 * 4 + 2) * 18 * 8) + threadIndexGlobal] = tempX[3][2];
            LDS_[((3 * 4 + 3) * 18 * 8) + threadIndexGlobal] = tempX[3][3];
        }
        else // threadIndex4x4 >= 8
        {
            uint threadIndexHigh = threadIndex4x4 & 7; // threadIndex4x4 - 8

            float4x4 tempW;
            uint tg_Dy;
            uint tg_Dx;
            [unroll] for (tg_Dy = 0; tg_Dy < BLOCK_SIZE; tg_Dy++)
            {
                [unroll] for (tg_Dx = 0; tg_Dx < BLOCK_SIZE; tg_Dx++)
                {
                    uint tg_KernelReadOffset = (tg_Dy * 4 + tg_Dx)*tg_KernelSpatialStride + tg_NumKernels * (tg_ChannelOffset + threadIndexHigh);
                    uint kernelReadOffset = tg_KernelReadOffset + tg_KernelBaseId + threadIndex;

#if LAX_KERNEL
                    kernelReadOffset = min(kernelReadOffset, K.GetLength() - 1);
#endif

                    tempW[tg_Dy][tg_Dx] = K.FastGet(kernelReadOffset);
                }
            }

            // store tempX interleaved per thread:
            // thread: 0 1 2 .... 128 0 1 2 .... 128 [16SGEMM x (8 values)]
            //         <- tempW[0] -> <- tempW[1] ->
            // to avoid bank conflict in the inner loop, we shift every tempW by 18*8 instead of 256=16*8
            // LDS_[W_OFFSET + ([0,15])*18*8 + ((threadIndexGlobal/16)-8)*16 + (threadIndexGlobal%16)] = tempX[[0,15]] // -8 to get (threadIndexGlobal/16) between 0,8
            // W_OFFSET = 15*18*8+7*16+15 + 1 = 2288
            LDS_[(2288 - 8*16 + (0 * 4 + 0) * 18 * 8) + threadIndexGlobal] = tempW[0][0];
            LDS_[(2288 - 8*16 + (0 * 4 + 1) * 18 * 8) + threadIndexGlobal] = tempW[0][1];
            LDS_[(2288 - 8*16 + (0 * 4 + 2) * 18 * 8) + threadIndexGlobal] = tempW[0][2];
            LDS_[(2288 - 8*16 + (0 * 4 + 3) * 18 * 8) + threadIndexGlobal] = tempW[0][3];
            LDS_[(2288 - 8*16 + (1 * 4 + 0) * 18 * 8) + threadIndexGlobal] = tempW[1][0];
            LDS_[(2288 - 8*16 + (1 * 4 + 1) * 18 * 8) + threadIndexGlobal] = tempW[1][1];
            LDS_[(2288 - 8*16 + (1 * 4 + 2) * 18 * 8) + threadIndexGlobal] = tempW[1][2];
            LDS_[(2288 - 8*16 + (1 * 4 + 3) * 18 * 8) + threadIndexGlobal] = tempW[1][3];
            LDS_[(2288 - 8*16 + (2 * 4 + 0) * 18 * 8) + threadIndexGlobal] = tempW[2][0];
            LDS_[(2288 - 8*16 + (2 * 4 + 1) * 18 * 8) + threadIndexGlobal] = tempW[2][1];
            LDS_[(2288 - 8*16 + (2 * 4 + 2) * 18 * 8) + threadIndexGlobal] = tempW[2][2];
            LDS_[(2288 - 8*16 + (2 * 4 + 3) * 18 * 8) + threadIndexGlobal] = tempW[2][3];
            LDS_[(2288 - 8*16 + (3 * 4 + 0) * 18 * 8) + threadIndexGlobal] = tempW[3][0];
            LDS_[(2288 - 8*16 + (3 * 4 + 1) * 18 * 8) + threadIndexGlobal] = tempW[3][1];
            LDS_[(2288 - 8*16 + (3 * 4 + 2) * 18 * 8) + threadIndexGlobal] = tempW[3][2];
            LDS_[(2288 - 8*16 + (3 * 4 + 3) * 18 * 8) + threadIndexGlobal] = tempW[3][3];
        }

        GroupMemoryBarrierWithGroupSync();

        // Inner loop
        //  uint ptrX =        ((threadIndexGlobal%16)/4)*4 + (threadIndexGlobal/16) * 18 * 8;
        //  uint ptrW = 2288 + ((threadIndexGlobal%16)%4)*4 + (threadIndexGlobal/16) * 18 * 8;
        uint ptrX = (groupThreadIDY4 + (threadIndex4x4 * 18 * 8));
        uint ptrW = 2288 + (groupThreadIDX4 + (threadIndex4x4 * 18 * 8));

        float colOfX[BLOCK_SIZE];
        float rowOfW[BLOCK_SIZE];

        [loop] for (uint tg_CacheExecuteIdx = 0; tg_CacheExecuteIdx < 8; ++tg_CacheExecuteIdx)
        {
            //Load LDS -> registers
            colOfX[0] = LDS_[ptrX | 0];
            colOfX[1] = LDS_[ptrX | 1];
            colOfX[2] = LDS_[ptrX | 2];
            colOfX[3] = LDS_[ptrX | 3];

            rowOfW[0] = LDS_[ptrW | 0];
            rowOfW[1] = LDS_[ptrW | 1];
            rowOfW[2] = LDS_[ptrW | 2];
            rowOfW[3] = LDS_[ptrW | 3];

            ptrX += 16;
            ptrW += 16;

            // Mads 4 pixels by 4 kernels matmul style --> 16 mads
            dstA[0*BLOCK_SIZE + 0] = ffma(colOfX[0], rowOfW[0], dstA[0*BLOCK_SIZE + 0]);
            dstA[0*BLOCK_SIZE + 1] = ffma(colOfX[0], rowOfW[1], dstA[0*BLOCK_SIZE + 1]);
            dstA[0*BLOCK_SIZE + 2] = ffma(colOfX[0], rowOfW[2], dstA[0*BLOCK_SIZE + 2]);
            dstA[0*BLOCK_SIZE + 3] = ffma(colOfX[0], rowOfW[3], dstA[0*BLOCK_SIZE + 3]);
            dstA[1*BLOCK_SIZE + 0] = ffma(colOfX[1], rowOfW[0], dstA[1*BLOCK_SIZE + 0]);
            dstA[1*BLOCK_SIZE + 1] = ffma(colOfX[1], rowOfW[1], dstA[1*BLOCK_SIZE + 1]);
            dstA[1*BLOCK_SIZE + 2] = ffma(colOfX[1], rowOfW[2], dstA[1*BLOCK_SIZE + 2]);
            dstA[1*BLOCK_SIZE + 3] = ffma(colOfX[1], rowOfW[3], dstA[1*BLOCK_SIZE + 3]);
            dstA[2*BLOCK_SIZE + 0] = ffma(colOfX[2], rowOfW[0], dstA[2*BLOCK_SIZE + 0]);
            dstA[2*BLOCK_SIZE + 1] = ffma(colOfX[2], rowOfW[1], dstA[2*BLOCK_SIZE + 1]);
            dstA[2*BLOCK_SIZE + 2] = ffma(colOfX[2], rowOfW[2], dstA[2*BLOCK_SIZE + 2]);
            dstA[2*BLOCK_SIZE + 3] = ffma(colOfX[2], rowOfW[3], dstA[2*BLOCK_SIZE + 3]);
            dstA[3*BLOCK_SIZE + 0] = ffma(colOfX[3], rowOfW[0], dstA[3*BLOCK_SIZE + 0]);
            dstA[3*BLOCK_SIZE + 1] = ffma(colOfX[3], rowOfW[1], dstA[3*BLOCK_SIZE + 1]);
            dstA[3*BLOCK_SIZE + 2] = ffma(colOfX[3], rowOfW[2], dstA[3*BLOCK_SIZE + 2]);
            dstA[3*BLOCK_SIZE + 3] = ffma(colOfX[3], rowOfW[3], dstA[3*BLOCK_SIZE + 3]);
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // store 16 SGEMM results to LDS
    // LDS_[(threadIndexGlobal/16)*16*17 + [0,15]*16 + threadIndexGlobal%16] = dstA[0,15]; 17 instead of 16 to avoid bank conflicts
    LDS_[threadIndex4x4 * 16 * 17 + 0   + threadIndex] = dstA[0];
    LDS_[threadIndex4x4 * 16 * 17 + 16  + threadIndex] = dstA[1];
    LDS_[threadIndex4x4 * 16 * 17 + 32  + threadIndex] = dstA[2];
    LDS_[threadIndex4x4 * 16 * 17 + 48  + threadIndex] = dstA[3];
    LDS_[threadIndex4x4 * 16 * 17 + 64  + threadIndex] = dstA[4];
    LDS_[threadIndex4x4 * 16 * 17 + 80  + threadIndex] = dstA[5];
    LDS_[threadIndex4x4 * 16 * 17 + 96  + threadIndex] = dstA[6];
    LDS_[threadIndex4x4 * 16 * 17 + 112 + threadIndex] = dstA[7];
    LDS_[threadIndex4x4 * 16 * 17 + 128 + threadIndex] = dstA[8];
    LDS_[threadIndex4x4 * 16 * 17 + 144 + threadIndex] = dstA[9];
    LDS_[threadIndex4x4 * 16 * 17 + 160 + threadIndex] = dstA[10];
    LDS_[threadIndex4x4 * 16 * 17 + 176 + threadIndex] = dstA[11];
    LDS_[threadIndex4x4 * 16 * 17 + 192 + threadIndex] = dstA[12];
    LDS_[threadIndex4x4 * 16 * 17 + 208 + threadIndex] = dstA[13];
    LDS_[threadIndex4x4 * 16 * 17 + 224 + threadIndex] = dstA[14];
    LDS_[threadIndex4x4 * 16 * 17 + 240 + threadIndex] = dstA[15];

    GroupMemoryBarrierWithGroupSync();

    // Load 4x4 accumulated result and perfom inverse winograd to get 2x2 output patch
    float4x4 tempY;
    // tempY[0,15] = LDS_[[0,15]*16*17 + (threadIndexGlobal/16) * 16 + threadIndex];
    tempY[0][0] = LDS_[ 0 * 16 * 17 + threadIndexGlobal];
    tempY[0][1] = LDS_[ 1 * 16 * 17 + threadIndexGlobal];
    tempY[0][2] = LDS_[ 2 * 16 * 17 + threadIndexGlobal];
    tempY[0][3] = LDS_[ 3 * 16 * 17 + threadIndexGlobal];
    tempY[1][0] = LDS_[ 4 * 16 * 17 + threadIndexGlobal];
    tempY[1][1] = LDS_[ 5 * 16 * 17 + threadIndexGlobal];
    tempY[1][2] = LDS_[ 6 * 16 * 17 + threadIndexGlobal];
    tempY[1][3] = LDS_[ 7 * 16 * 17 + threadIndexGlobal];
    tempY[2][0] = LDS_[ 8 * 16 * 17 + threadIndexGlobal];
    tempY[2][1] = LDS_[ 9 * 16 * 17 + threadIndexGlobal];
    tempY[2][2] = LDS_[10 * 16 * 17 + threadIndexGlobal];
    tempY[2][3] = LDS_[11 * 16 * 17 + threadIndexGlobal];
    tempY[3][0] = LDS_[12 * 16 * 17 + threadIndexGlobal];
    tempY[3][1] = LDS_[13 * 16 * 17 + threadIndexGlobal];
    tempY[3][2] = LDS_[14 * 16 * 17 + threadIndexGlobal];
    tempY[3][3] = LDS_[15 * 16 * 17 + threadIndexGlobal];

    float2x2 writeValue = ApplyWinnogradA(tempY);

    // store 2x2 patch to have coalesced writes
    GroupMemoryBarrierWithGroupSync();

    // LDS_[[0,3]*(3*77+3*16+3*4+3+1) + ((threadIndexGlobal/16)/4)*77 + ((threadIndexGlobal/16)%4)*16 + ((threadIndexGlobal%16)/4)*4 + ((threadIndexGlobal%16)%4)] = writeValue[0,3]; // 77 instead of 64 to avoid bank conflicts
    LDS_[0*(295) + (threadIndex4x4 >> 2)*77 + (((threadIndex4x4 & 0x3) << 4) | threadIndex)] = writeValue[0][0];
    LDS_[1*(295) + (threadIndex4x4 >> 2)*77 + (((threadIndex4x4 & 0x3) << 4) | threadIndex)] = writeValue[0][1];
    LDS_[2*(295) + (threadIndex4x4 >> 2)*77 + (((threadIndex4x4 & 0x3) << 4) | threadIndex)] = writeValue[1][0];
    LDS_[3*(295) + (threadIndex4x4 >> 2)*77 + (((threadIndex4x4 & 0x3) << 4) | threadIndex)] = writeValue[1][1];
    GroupMemoryBarrierWithGroupSync();

    // writeValue[[0,3]] = LDS_[[0,3]*(3*77+3*16+3*4+3+1) + ((threadIndexGlobal%16)%4)*77 + ((threadIndexGlobal/16)%4)*16 + ((threadIndexGlobal%16)/4)*4 + ((threadIndexGlobal/16)/4)];
    writeValue[0][0] = LDS_[0*(295) + (threadIndex & 0x3)*77 + (((threadIndex4x4 & 0x3) << 4) | groupThreadIDY4 | (threadIndex4x4 >> 2))];
    writeValue[0][1] = LDS_[1*(295) + (threadIndex & 0x3)*77 + (((threadIndex4x4 & 0x3) << 4) | groupThreadIDY4 | (threadIndex4x4 >> 2))];
    writeValue[1][0] = LDS_[2*(295) + (threadIndex & 0x3)*77 + (((threadIndex4x4 & 0x3) << 4) | groupThreadIDY4 | (threadIndex4x4 >> 2))];
    writeValue[1][1] = LDS_[3*(295) + (threadIndex & 0x3)*77 + (((threadIndex4x4 & 0x3) << 4) | groupThreadIDY4 | (threadIndex4x4 >> 2))];


    uint writeChannelId = tg_KernelBaseId + threadIndex4x4;
    uint writePixelId = tg_OutputPixelBaseId + threadIndex;

    writeValue += B.FastGet(min(tg_NumKernels-1, writeChannelId));

    uint2 writePixelCoords = 2 * int2(writePixelId % tg_WidthOHalf, writePixelId / tg_WidthOHalf);

#if LAX_KERNEL
    bool canWriteChannel = (writeChannelId < tg_NumKernels);
#else
    bool canWriteChannel = true;
#endif

    uint writeIndex = O.width * O.height * writeChannelId + tg_BatchWriteOffset;

    if (canWriteChannel && writePixelCoords.y < tg_HeightO && writePixelCoords.x < tg_WidthO)
        O.FastSetWithActivation(writeIndex + (writePixelCoords.y) * tg_WidthO + (writePixelCoords.x), writeValue[0][0]);
    if (canWriteChannel && writePixelCoords.y < tg_HeightO && (writePixelCoords.x + 1) < tg_WidthO)
        O.FastSetWithActivation(writeIndex + (writePixelCoords.y) * tg_WidthO + (writePixelCoords.x + 1), writeValue[0][1]);
    if (canWriteChannel && (writePixelCoords.y + 1) < tg_HeightO && writePixelCoords.x < tg_WidthO)
        O.FastSetWithActivation(writeIndex + (writePixelCoords.y + 1) * tg_WidthO + (writePixelCoords.x), writeValue[1][0]);
    if (canWriteChannel && (writePixelCoords.y + 1) < tg_HeightO && (writePixelCoords.x + 1) < tg_WidthO)
        O.FastSetWithActivation(writeIndex + (writePixelCoords.y + 1) * tg_WidthO + (writePixelCoords.x + 1), writeValue[1][1]);


#undef X_OFFSET
#undef W_OFFSET
#undef LDS_
#undef X_
#undef W_
}
#undef CACHE_DEPTH
#undef CACHE_WIDTH
#undef SHUFFLE_FOR_COALESCED_LOAD
#undef SHUFFLE_FOR_COALESCED_STORE
#undef _PAD
#undef CACHE_DEPTH
#undef PIXELS_PER_CACHE
#undef NUMTHREADS_PER_TG
#undef SHUFFLE_FOR_COALESCED_LOAD
#undef SHUFFLE_FOR_COALESCED_STORE
#endif //KERNEL_PER_TG == 16
#endif //BLOCK_SIZE == 4
