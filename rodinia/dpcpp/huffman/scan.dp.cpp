/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "scanLargeArray_kernel.dp.cpp"
#include <assert.h>
#include <stdio.h>
#include "cutil.h"
#include <cmath>

#include <algorithm>

inline bool
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

#define BLOCK_SIZE 256

static unsigned int** g_scanBlockSums;
static unsigned int g_numEltsAllocated = 0;
static unsigned int g_numLevelsAllocated = 0;

static void preallocBlockSums(unsigned int maxNumElements) try {
    assert(g_numEltsAllocated == 0); // shouldn't be called 

    g_numEltsAllocated = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;
    int level = 0;

    do {
        unsigned int numBlocks =
            std::max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) level++;
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (unsigned int**) malloc(level * sizeof(unsigned int*));
    g_numLevelsAllocated = level;
    numElts = maxNumElements;
    level = 0;
    
    do {
        unsigned int numBlocks =
            std::max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
            /*
            DPCT1003:0: Migrated API does not return error code.
             * (*, 0) is inserted. You may need to rewrite this code.
 */
            CUDA_SAFE_CALL(
                (g_scanBlockSums[level++] = sycl::malloc_device<unsigned int>(
                     numBlocks, dpct::get_default_queue()),
                 0));
        numElts = numBlocks;
    } while (numElts > 1);

    CUT_CHECK_ERROR("preallocBlockSums");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

static void deallocBlockSums()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated; i++)
    {
        sycl::free(g_scanBlockSums[i], dpct::get_default_queue());
    }

    CUT_CHECK_ERROR("deallocBlockSums");
    
    free((void**)g_scanBlockSums);

    g_scanBlockSums = 0;
    g_numEltsAllocated = 0;
    g_numLevelsAllocated = 0;
}

static void prescanArrayRecursive(unsigned int *outArray, 
                           const unsigned int *inArray, 
                           int numElements, 
                           int level)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks =
        std::max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock =
        std::max<unsigned int>(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = 
            sizeof(unsigned int) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = 
        sizeof(unsigned int) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltsAllocated >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    sycl::range<3> grid(1, 1,
                        std::max<unsigned int>(1, numBlocks - np2LastBlock));
    sycl::range<3> threads(1, 1, numThreads);

    // make sure there are no CUDA errors before we start
    CUT_CHECK_ERROR("prescanArrayRecursive before kernels");

    // execute the scan
    if (numBlocks > 1)
    {
        /*
        DPCT1049:4: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
        */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      dpct_local_acc_ct1(sycl::range<1>(sharedMemSize), cgh);

                  auto g_scanBlockSums_level_ct2 = g_scanBlockSums[level];

                  cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         prescan<true, false>(
                                             outArray, inArray,
                                             g_scanBlockSums_level_ct2,
                                             numThreads * 2, 0, 0, item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                   });
            });
        CUT_CHECK_ERROR("prescanWithBlockSums");
        if (np2LastBlock)
        {
            /*
            DPCT1049:6: The workgroup size passed to the SYCL
             * kernel may exceed the limit. To get the device limit, query
             * info::device::max_work_group_size. Adjust the workgroup size if
             * needed.
            */
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::accessor<uint8_t, 1,
                                       sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            dpct_local_acc_ct1(
                                sycl::range<1>(sharedMemLastBlock), cgh);

                        auto g_scanBlockSums_level_ct2 = g_scanBlockSums[level];

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, numThreadsLastBlock),
                                sycl::range<3>(1, 1, numThreadsLastBlock)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  prescan<true, true>(
                                      outArray, inArray,
                                      g_scanBlockSums_level_ct2,
                                      numEltsLastBlock, numBlocks - 1,
                                      numElements - numEltsLastBlock, item_ct1,
                                      dpct_local_acc_ct1.get_pointer());
                            });
                  });
            CUT_CHECK_ERROR("prescanNP2WithBlockSums");
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1);

        /*
        DPCT1049:5: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
        */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::accessor<unsigned int, 0,
                                 sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      uni_acc_ct1(cgh);

                  auto g_scanBlockSums_level_ct1 = g_scanBlockSums[level];

                  cgh.parallel_for(
                      sycl::nd_range<3>(grid * threads, threads),
                      [=](sycl::nd_item<3> item_ct1) {
                            uniformAdd(outArray, g_scanBlockSums_level_ct1,
                                       numElements - numEltsLastBlock, 0, 0,
                                       item_ct1, uni_acc_ct1.get_pointer());
                      });
            });
        CUT_CHECK_ERROR("uniformAdd");
        if (np2LastBlock)
        {
            /*
            DPCT1049:7: The workgroup size passed to the SYCL
             * kernel may exceed the limit. To get the device limit, query
             * info::device::max_work_group_size. Adjust the workgroup size if
             * needed.
            */
                  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        sycl::accessor<unsigned int, 0,
                                       sycl::access::mode::read_write,
                                       sycl::access::target::local>
                            uni_acc_ct1(cgh);

                        auto g_scanBlockSums_level_ct1 = g_scanBlockSums[level];

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, numThreadsLastBlock),
                                sycl::range<3>(1, 1, numThreadsLastBlock)),
                            [=](sycl::nd_item<3> item_ct1) {
                                  uniformAdd(
                                      outArray, g_scanBlockSums_level_ct1,
                                      numEltsLastBlock, numBlocks - 1,
                                      numElements - numEltsLastBlock, item_ct1,
                                      uni_acc_ct1.get_pointer());
                            });
                  });
            CUT_CHECK_ERROR("uniformAdd");
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        /*
        DPCT1049:8: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
        */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      dpct_local_acc_ct1(sycl::range<1>(sharedMemSize), cgh);

                  cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         prescan<false, false>(
                                             outArray, inArray, 0,
                                             numThreads * 2, 0, 0, item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                   });
            });
        CUT_CHECK_ERROR("prescan");
    }
    else
    {
         /*
         DPCT1049:9: The workgroup size passed to the SYCL kernel
          * may exceed the limit. To get the device limit, query
          * info::device::max_work_group_size. Adjust the workgroup size if
          * needed.
         */
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                  sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                                 sycl::access::target::local>
                      dpct_local_acc_ct1(sycl::range<1>(sharedMemSize), cgh);

                  cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                   [=](sycl::nd_item<3> item_ct1) {
                                         prescan<false, true>(
                                             outArray, inArray, 0, numElements,
                                             0, 0, item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                   });
            });
         CUT_CHECK_ERROR("prescanNP2");
    }
}

static void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{
    prescanArrayRecursive(outArray, inArray, numElements, 0);
}

#endif // _PRESCAN_CU_
