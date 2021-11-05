/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA
 *
 * Copyright (C) 2009 Tjark Bringewat <golvellius@gmx.net>, Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 * 
 */


#ifndef _PACK_KERNELS_H_
#define _PACK_KERNELS_H_
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "parameters.h"

static void pack2(unsigned int *srcData, unsigned int *cindex, unsigned int *cindex2, unsigned int *dstData, unsigned int original_num_block_elements,
                  sycl::nd_item<3> item_ct1) {
        unsigned int tid =
            item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);

        // source index
	unsigned int offset = tid * original_num_block_elements;//DPB,
	unsigned int bitsize = cindex[tid];

	// destination index
	unsigned int pos = cindex2[tid],
				 dword = pos / 32,
				 bit = pos % 32;

	unsigned int i, dw, tmp;
	dw = srcData[offset];			// load the first dword from srcData[]
	tmp = dw >> bit;				// cut off those bits that do not fit into the initial location in destData[]
        /*
	DPCT1039:30: The generated code assumes that "&dstData[dword]"
         * points to the global memory address space. If it points to a local
         * memory address space, replace "dpct::atomic_fetch_or" with
         * "dpct::atomic_fetch_or<unsigned int,
         * sycl::access::address_space::local_space>".
	*/
        sycl::atomic<unsigned int>(
            sycl::global_ptr<unsigned int>(&dstData[dword]))
            .fetch_or(tmp); // fill up this initial location
        tmp = dw << 32-bit;				// save the remaining bits that were cut off earlier in tmp
	for (i=1; i<bitsize/32; i++) {	// from now on, we have exclusive access to destData[]
		dw = srcData[offset+i];		// load next dword from srcData[]
		tmp |= dw >> bit;			// fill up tmp
		dstData[dword+i] = tmp;		// write complete dword to destData[]
		tmp = dw << 32-bit;			// save the remaining bits in tmp (like before)
	}
	// exclusive access to dstData[] ends here
	// the remaining block can, or rather should be further optimized
	// write the remaining bits in tmp, UNLESS bit is 0 and bitsize is divisible by 32, in this case do nothing
	if (bit != 0 || bitsize % 32 != 0)
                /*
		DPCT1039:31: The generated code assumes that
                 * "&dstData[dword+i]" points to the global memory address
                 * space. If it points to a local memory address space, replace
                 * "dpct::atomic_fetch_or" with "dpct::atomic_fetch_or<unsigned
                 * int, sycl::access::address_space::local_space>".

                 */
                sycl::atomic<unsigned int>(
                    sycl::global_ptr<unsigned int>(&dstData[dword + i]))
                    .fetch_or(tmp);
        if (bitsize % 32 != 0) {
		dw = srcData[offset+i];
                /*
		DPCT1039:32: The generated code assumes that
                 * "&dstData[dword+i]" points to the global memory address
                 * space. If it points to a local memory address space, replace
                 * "dpct::atomic_fetch_or" with "dpct::atomic_fetch_or<unsigned
                 * int, sycl::access::address_space::local_space>".

                 */
                sycl::atomic<unsigned int>(
                    sycl::global_ptr<unsigned int>(&dstData[dword + i]))
                    .fetch_or(dw >> bit);
                /*
		DPCT1039:33: The generated code assumes that
                 * "&dstData[dword+i+1]" points to the global memory address
                 * space. If it points to a local memory address space, replace
                 * "dpct::atomic_fetch_or" with "dpct::atomic_fetch_or<unsigned
                 * int, sycl::access::address_space::local_space>".

                 */
                sycl::atomic<unsigned int>(
                    sycl::global_ptr<unsigned int>(&dstData[dword + i + 1]))
                    .fetch_or(dw << 32 - bit);
        }
}

#endif
