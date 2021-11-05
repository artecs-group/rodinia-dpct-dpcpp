/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA; S
 * Huffman encoding when using tree generation with restriction on the size (max. Huffman codeword dize = 2x original symbol size).
 *
 * Copyright (C) 2009 Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 * 
 */

#ifndef _VLC_SM64HUFF_KERNEL_H_
#define _VLC_SM64HUFF_KERNEL_H_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "parameters.h"
#include "pabio_kernels_v2.dp.cpp"

#ifdef SMATOMICS

/* HUFFMAN-FRIENDLY PAVLE
   CHARACTERISTICS:
   1. CACHE CW_LUT INTO SM, LOAD AS 2 INT ARRAYS
   2. PARALLEL PREFIX SUM
   3. PARALLEL BIT I/O USING SHARED-MEMORY ATOMIC OPERATIONS (COMAPTIBLE WITH CUDA1.3+)
   
   NOTES & ASSUMPTIONS:
   -	HUFFMAN-CODING FRIENDLY, SUPPORTS CODEWORDS OF 2X SIZE OF ORIGINAL SYMBOLS (BYTES). 
   -	NUMBER OF THREADS PER BLOCK IS 256; IF YOU WANT TO PLAY WITH DIFFERENT NUMBERS, THE CW CACHING SHOULD BE MODIFIED (SEE DPT* KERNELS) 
   -	SM usage: 1x size of the input data (REUSE) + size of CWLUT
		TURN ON CACHING FOR HIGH ENTROPY DATA!
*/


static void vlc_encode_kernel_sm64huff(unsigned int* data,
								  const unsigned int* gm_codewords, const unsigned int* gm_codewordlens,
							#ifdef TESTING
								  unsigned int* cw32, unsigned int* cw32len, unsigned int* cw32idx, 
							#endif
								  unsigned int* out, unsigned int *outidx, sycl::nd_item<3> item_ct1,
								  uint8_t *dpct_local, unsigned int *kcmax){

        unsigned int kn =
            item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
        unsigned int k = item_ct1.get_local_id(2);
        unsigned int kc, startbit, wrbits;

	unsigned long long cw64 =0;
	unsigned int val32, codewordlen = 0;
	unsigned char tmpbyte, tmpcwlen;
	unsigned int tmpcw32;

        auto sm = (unsigned int *)dpct_local;

#ifdef CACHECWLUT
	unsigned int* codewords		= (unsigned int*) sm; 
	unsigned int* codewordlens	= (unsigned int*)(sm+NUM_SYMBOLS); 
	unsigned int* as			= (unsigned int*)(sm+2*NUM_SYMBOLS);

	/* Load the codewords and the original data*/
	codewords[k]	= gm_codewords[k];
	codewordlens[k] = gm_codewordlens[k];
	val32			= data[kn];
        item_ct1.barrier();
        for(unsigned int i=0; i<4;i++) {
		tmpbyte = (unsigned char)(val32>>((3-i)*8));
		tmpcw32 = codewords[tmpbyte];
		tmpcwlen = codewordlens[tmpbyte];
		cw64 = (cw64<<tmpcwlen) | tmpcw32;
		codewordlen+=tmpcwlen;
	}
#else
	unsigned int* as			= (unsigned int*) sm;
	val32 = data[kn];
	for(unsigned int i=0; i<4;i++) {
		tmpbyte = (unsigned char)(val32>>((3-i)*8));
		tmpcw32 = gm_codewords[tmpbyte];
		tmpcwlen = gm_codewordlens[tmpbyte];
		cw64 = (cw64<<tmpcwlen) | tmpcw32;
		codewordlen+=tmpcwlen;
	}
#endif
	as[k] = codewordlen;
        item_ct1.barrier();

        /* Prefix sum of codeword lengths (denoted in bits) [inplace implementation] */ 
	unsigned int offset = 1;

    /* Build the sum in place up the tree */
    for (unsigned int d = (item_ct1.get_local_range().get(2)) >> 1; d > 0;
         d >>= 1) {
        item_ct1.barrier();
        if (k < d)   {
            unsigned char ai = offset*(2*k+1)-1;
            unsigned char bi = offset*(2*k+2)-1;
            as[bi] += as[ai];
        }
        offset *= 2;
    }

    /* scan back down the tree */
    /* clear the last element */
    if (k == 0) as[item_ct1.get_local_range().get(2) - 1] = 0;

    // traverse down the tree building the scan in place
    for (unsigned int d = 1; d < item_ct1.get_local_range().get(2); d *= 2) {
        offset >>= 1;
        item_ct1.barrier();
        if (k < d)   {
            unsigned char ai = offset*(2*k+1)-1;
            unsigned char bi = offset*(2*k+2)-1;
            unsigned int t   = as[ai];
            as[ai]  = as[bi];
            as[bi] += t;
        }
    }
        item_ct1.barrier();

        if (k == item_ct1.get_local_range().get(2) - 1) {
                outidx[item_ct1.get_group(2)] = as[k] + codewordlen;
                *kcmax = (as[k] + codewordlen) / 32;
        }

	/* Write the codes */
	kc = as[k]/32; 
	startbit = as[k]%32; 
	as[k] =  0U;
        item_ct1.barrier();

        /* Part 1*/
	wrbits		= codewordlen > (32-startbit)? (32-startbit): codewordlen;
	tmpcw32		= (unsigned int)(cw64>>(codewordlen - wrbits));		
	//if (wrbits == 32) as[kc] = tmpcw32;				//unnecessary overhead; increases number of branches
	//else
        sycl::atomic<unsigned int, sycl::access::address_space::local_space>(
            sycl::local_ptr<unsigned int>(&as[kc]))
            .fetch_or(tmpcw32 << (32 - startbit -
                                  wrbits)); // shift left in case it's shorter
                                            // then the available space
        codewordlen		-= wrbits;

	/*Part 2*/
	if (codewordlen) {
	wrbits		= codewordlen > 32 ? 32: codewordlen;
	tmpcw32		= (unsigned int)(cw64>>(codewordlen - wrbits)) & ((1<<wrbits)-1);	
	//if (wrbits == 32) as[kc+1] = tmpcw32;
	//else
        sycl::atomic<unsigned int, sycl::access::address_space::local_space>(
            sycl::local_ptr<unsigned int>(&as[kc + 1]))
            .fetch_or(tmpcw32 << (32 - wrbits));
        codewordlen	-= wrbits;
	}

	/*Part 3*/
	if (codewordlen) {
	tmpcw32		= (unsigned int)(cw64 & ((1<<codewordlen)-1));
	//if (wrbits == 32) as[kc+2] = tmpcw32;
	//else
        sycl::atomic<unsigned int, sycl::access::address_space::local_space>(
            sycl::local_ptr<unsigned int>(&as[kc + 2]))
            .fetch_or(tmpcw32 << (32 - codewordlen));
        }

        item_ct1.barrier();

        if (k <= *kcmax) out[kn] = as[k];
}
//////////////////////////////////////////////////////////////////////////////								  
#endif

#endif