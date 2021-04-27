#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
// statistical kernel
void extract(	long d_Ne,
											fp *d_I,
											sycl::nd_item<3> item_ct1){										// pointer to input image (DEVICE GLOBAL MEMORY)

	// indexes
        int bx = item_ct1.get_group(2);    // get current horizontal block index
        int tx = item_ct1.get_local_id(2); // get current horizontal thread index
        int ei = (bx*NUMBER_THREADS)+tx;						// unique thread id, more threads than actual elements !!!

	// copy input to output & log uncompress
	if(ei<d_Ne){															// do only for the number of elements, omit extra threads

                d_I[ei] = sycl::exp(
                    d_I[ei] /
                    255); // exponentiate input IMAGE and copy to output image
        }

}
