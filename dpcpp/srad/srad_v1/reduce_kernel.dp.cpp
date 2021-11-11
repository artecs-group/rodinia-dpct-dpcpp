#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
// statistical kernel
void reduce(	long d_Ne,											// number of elements in array
										int d_no,											// number of sums to reduce
										int d_mul,											// increment
										fp *d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
										fp *d_sums2,
										sycl::nd_item<3> item_ct1,
										fp *d_psum,
										fp *d_psum2, int NUMBER_THREADS){

	// indexes
    int bx = item_ct1.get_group(2);    // get current horizontal block index
        int tx = item_ct1.get_local_id(2); // get current horizontal thread index
        int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!
        int nf = NUMBER_THREADS -
                 (item_ct1.get_group_range(2) * NUMBER_THREADS -
                  d_no); // number of elements assigned to last block
        int df = 0;																// divisibility factor for the last block

        // statistical
        // data for block calculations allocated by every block in its shared
        // memory

        // counters
	int i;

	// copy data to shared memory
	if(ei<d_no){															// do only for the number of elements, omit extra threads

		d_psum[tx] = d_sums[ei*d_mul];
		d_psum2[tx] = d_sums2[ei*d_mul];

	}

	// Lingjie Zhang modifited at Nov 1 / 2015
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
     * sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
     * performance, if there is no access to global memory.
    */
    item_ct1.barrier();
    // end Lingjie Zhang's modification

	// reduction of sums if all blocks are full (rare case)	
	if(nf == NUMBER_THREADS){
		// sum of every 2, 4, ..., NUMBER_THREADS elements
		for(i=2; i<=NUMBER_THREADS; i=2*i){
			// sum of elements
			if((tx+1) % i == 0){											// every ith
				d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
				d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
			}
			// synchronization
                        /*
			DPCT1065:1: Consider replacing
                         * sycl::nd_item::barrier() with
                         * sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                         * for better performance, if there is no access to
                         * global memory.
			*/
                        item_ct1.barrier();
                }
		// final sumation by last thread in every block
		if(tx==(NUMBER_THREADS-1)){											// block result stored in global memory
			d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
			d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
		}
	}
	// reduction of sums if last block is not full (common case)
	else{ 
		// for full blocks (all except for last block)
                if (bx != (item_ct1.get_group_range(2) - 1)) { //
                        // sum of every 2, 4, ..., NUMBER_THREADS elements
			for(i=2; i<=NUMBER_THREADS; i=2*i){								//
				// sum of elements
				if((tx+1) % i == 0){										// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization
                                /*
				DPCT1065:2: Consider
                                 * replacing sycl::nd_item::barrier() with
                                 * sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                                 * for better performance, if there is no access
                                 * to global memory.
 */
                                item_ct1.barrier(); //
                        }
			// final sumation by last thread in every block
			if(tx==(NUMBER_THREADS-1)){										// block result stored in global memory
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
		}
		// for not full block (last block)
		else{																//
			// figure out divisibility
			for(i=2; i<=NUMBER_THREADS; i=2*i){								//
				if(nf >= i){
					df = i;
				}
			}
			// sum of every 2, 4, ..., NUMBER_THREADS elements
			for(i=2; i<=df; i=2*i){											//
				// sum of elements (only busy threads)
				if((tx+1) % i == 0 && tx<df){								// every ith
					d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
					d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
				}
				// synchronization (all threads)
                                /*
				DPCT1065:3: Consider
                                 * replacing sycl::nd_item::barrier() with
                                 * sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                                 * for better performance, if there is no access
                                 * to global memory.
 */
                                item_ct1.barrier(); //
                        }
			// remainder / final summation by last thread
			if(tx==(df-1)){										//
				// compute the remainder and final summation by last busy thread
				for(i=(bx*NUMBER_THREADS)+df; i<(bx*NUMBER_THREADS)+nf; i++){						//
					d_psum[tx] = d_psum[tx] + d_sums[i];
					d_psum2[tx] = d_psum2[tx] + d_sums2[i];
				}
				// final sumation by last thread in every block
				d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
				d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
			}
		}
	}

}
