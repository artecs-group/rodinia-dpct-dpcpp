#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
// statistical kernel
void prepare(	long d_Ne,
											fp *d_I,										// pointer to output image (DEVICE GLOBAL MEMORY)
											fp *d_sums,									// pointer to input image (DEVICE GLOBAL MEMORY)
											fp *d_sums2,
											sycl::nd_item<3> item_ct1){

	// indexes
        int bx = item_ct1.get_group(2);    // get current horizontal block index
        int tx = item_ct1.get_local_id(2); // get current horizontal thread index
        int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

		d_sums[ei] = d_I[ei];
		d_sums2[ei] = d_I[ei]*d_I[ei];

	}

}
