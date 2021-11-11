#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "track_ellipse_kernel.h"
#include "misc_math.h"
// #include <cutil.h>

// Constants used in the MGVF computation
#define ONE_OVER_PI (1.0 / PI)
#define MU 0.5
#define LAMBDA (8.0 * MU + 1.0)


// Host and device arrays to hold device pointers to input matrices
float **host_I_array, **host_IMGVF_array;
float **device_I_array, **device_IMGVF_array;
// Host and device arrays to hold sizes of input matrices
int *host_m_array, *host_n_array;
int *device_m_array, *device_n_array;

// Host array to hold matrices for all cells
// (so we can copy to and from the device in a single transfer)
float *host_I_all;
int total_mem_size;

// The number of threads per thread block
const int threads_per_block = 320;
// next_lowest_power_of_two = 2^(floor(log2(threads_per_block)))
const int next_lowest_power_of_two = 256;


// Regularized version of the Heaviside step function:
// He(x) = (atan(x) / pi) + 0.5
float heaviside(float x) {
        return (sycl::atan(x) * ONE_OVER_PI) + 0.5;

        // A simpler, faster approximation of the Heaviside function
	/* float out = 0.0;
	if (x > -0.0001) out = 0.5;
	if (x >  0.0001) out = 1.0;
	return out; */
}


// Kernel to compute the Motion Gradient Vector Field (MGVF) matrix for multiple cells
void IMGVF_kernel(float **IMGVF_array, float **I_array, int *m_array, int *n_array,
							 float vx, float vy, float e, int max_iterations, float cutoff,
							 sycl::nd_item<3> item_ct1, float *IMGVF, float *buffer,
							 int *cell_converged) {
	
	// Shared copy of the matrix being computed

        // Shared buffer used for two purposes:
	// 1) To temporarily store newly computed matrix values so that only
	//    values from the previous iteration are used in the computation.
	// 2) To store partial sums during the tree reduction which is performed
	//    at the end of each iteration to determine if the computation has converged.

        // Figure out which cell this thread block is working on
        int cell_num = item_ct1.get_group(2);

        // Get pointers to current cell's input image and inital matrix
	float *IMGVF_global = IMGVF_array[cell_num];
	float *I = I_array[cell_num];
	
	// Get current cell's matrix dimensions
	int m = m_array[cell_num];
	int n = n_array[cell_num];
	
	// Compute the number of virtual thread blocks
	int max = (m * n + threads_per_block - 1) / threads_per_block;
	
	// Load the initial IMGVF matrix into shared memory
        int thread_id = item_ct1.get_local_id(2), thread_block, i, j;
        for (thread_block = 0; thread_block < max; thread_block++) {
		int offset = thread_block * threads_per_block;
		i = (thread_id + offset) / n;
		j = (thread_id + offset) % n;
		if (i < m) IMGVF[(i * n) + j] = IMGVF_global[(i * n) + j];
	}
        /*
        DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance, if there is no access to global memory.
        */
        item_ct1.barrier();

        // Set the converged flag to false

        if (item_ct1.get_local_id(2) == 0) *cell_converged = 0;
        /*
        DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance, if there is no access to global memory.
        */
        item_ct1.barrier();

        // Constants used to iterate through virtual thread blocks
	const float one_nth = 1.f / (float) n;
	const int tid_mod = thread_id % n;
	const int tbsize_mod = threads_per_block % n;
	
	// Constant used in the computation of Heaviside values
	float one_over_e = 1.0 / e;
	
	// Iteratively compute the IMGVF matrix until the computation has
	//  converged or we have reached the maximum number of iterations
	int iterations = 0;
        while ((!(*cell_converged)) && (iterations < max_iterations)) {

                // The total change to this thread's matrix elements in the current iteration
		float total_diff = 0.0f;
		
		int old_i = 0, old_j = 0;
		j = tid_mod - tbsize_mod;
		
		// Iterate over virtual thread blocks
		for (thread_block = 0; thread_block < max; thread_block++) {
			// Store the index of this thread's previous matrix element
			//  (used in the buffering scheme below)
			old_i = i;
			old_j = j;
			
			// Determine the index of this thread's current matrix element 
			int offset = thread_block * threads_per_block;
			i = (thread_id + offset) * one_nth;
			j += tbsize_mod;
			if (j >= n) j -= n;
			
			float new_val = 0.0, old_val = 0.0;
			
			// Make sure the thread has not gone off the end of the matrix
			if (i < m) {
				// Compute neighboring matrix element indices
				int rowU = (i == 0) ? 0 : i - 1;
				int rowD = (i == m - 1) ? m - 1 : i + 1;
				int colL = (j == 0) ? 0 : j - 1;
				int colR = (j == n - 1) ? n - 1 : j + 1;
				
				// Compute the difference between the matrix element and its eight neighbors
				old_val = IMGVF[(i * n) + j];
				float U  = IMGVF[(rowU * n) + j   ] - old_val;
				float D  = IMGVF[(rowD * n) + j   ] - old_val;
				float L  = IMGVF[(i    * n) + colL] - old_val;
				float R  = IMGVF[(i    * n) + colR] - old_val;
				float UR = IMGVF[(rowU * n) + colR] - old_val;
				float DR = IMGVF[(rowD * n) + colR] - old_val;
				float UL = IMGVF[(rowU * n) + colL] - old_val;
				float DL = IMGVF[(rowD * n) + colL] - old_val;
				
				// Compute the regularized heaviside value for these differences
				float UHe  = heaviside((U  *       -vy)  * one_over_e);
				float DHe  = heaviside((D  *        vy)  * one_over_e);
				float LHe  = heaviside((L  *  -vx     )  * one_over_e);
				float RHe  = heaviside((R  *   vx     )  * one_over_e);
				float URHe = heaviside((UR * ( vx - vy)) * one_over_e);
				float DRHe = heaviside((DR * ( vx + vy)) * one_over_e);
				float ULHe = heaviside((UL * (-vx - vy)) * one_over_e);
				float DLHe = heaviside((DL * (-vx + vy)) * one_over_e);
				
				// Update the IMGVF value in two steps:
				// 1) Compute IMGVF += (mu / lambda)(UHe .*U  + DHe .*D  + LHe .*L  + RHe .*R +
				//                                   URHe.*UR + DRHe.*DR + ULHe.*UL + DLHe.*DL);
				new_val = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
													 URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
				// 2) Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
				float vI = I[(i * n) + j];
				new_val -= ((1.0 / LAMBDA) * vI * (new_val - vI));
			}
			
			// Save the previous virtual thread block's value (if it exists)
			if (thread_block > 0) {
				offset = (thread_block - 1) * threads_per_block;
				if (old_i < m) IMGVF[(old_i * n) + old_j] = buffer[thread_id];
			}
			if (thread_block < max - 1) {
				// Write the new value to the buffer
				buffer[thread_id] = new_val;
			} else {
				// We've reached the final virtual thread block,
				//  so write directly to the matrix
				if (i < m) IMGVF[(i * n) + j] = new_val;
			}
			
			// Keep track of the total change of this thread's matrix elements
                        total_diff += sycl::fabs(new_val - old_val);

                        // We need to synchronize between virtual thread blocks to prevent
			//  threads from writing the values from the buffer to the actual
			//  IMGVF matrix too early
                        /*
                        DPCT1065:14: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance, if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();
                }
		
		// We need to compute the overall sum of the change at each matrix element
		//  by performing a tree reduction across the whole threadblock
		buffer[thread_id] = total_diff;
                /*
                DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance, if there is no access to global memory.
                */
                item_ct1.barrier();

                // Account for thread block sizes that are not a power of 2
		if (thread_id >= next_lowest_power_of_two) {
			buffer[thread_id - next_lowest_power_of_two] += buffer[thread_id];
		}
                /*
                DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance, if there is no access to global memory.
                */
                item_ct1.barrier();

                // Perform the tree reduction
		int th;
		for (th = next_lowest_power_of_two / 2; th > 0; th /= 2) {
			if (thread_id < th) {
				buffer[thread_id] += buffer[thread_id + th];
			}
                        /*
                        DPCT1065:15: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance, if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();
                }
		
		// Figure out if we have converged
		if(thread_id == 0) {
			float mean = buffer[thread_id] / (float) (m * n);
			if (mean < cutoff) {
				// We have converged, so set the appropriate flag
                                *cell_converged = 1;
                        }
		}
		
		// We need to synchronize to ensure that all threads
		//  read the correct value of the convergence flag
                /*
                DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance, if there is no access to global memory.
                */
                item_ct1.barrier();

                // Keep track of the number of iterations we have performed
		iterations++;
	}
	
	// Save the final IMGVF matrix to global memory
	for (thread_block = 0; thread_block < max; thread_block++) {
		int offset = thread_block * threads_per_block;
		i = (thread_id + offset) / n;
		j = (thread_id + offset) % n;
		if (i < m) IMGVF_global[(i * n) + j] = IMGVF[(i * n) + j];
	}
}


// Host function that launches a CUDA kernel to compute the MGVF matrices for the specified cells
void IMGVF_cuda(MAT **I, MAT **IMGVF, double vx, double vy, double e, int max_iterations, double cutoff, int num_cells) {
	
	// Initialize the data on the GPU
	IMGVF_cuda_init(I, num_cells);
	
	// Compute the MGVF on the GPU
        /*
        DPCT1049:16: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor<float, 1, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    IMGVF_acc_ct1(sycl::range<1>(3321 /*41 * 81*/), cgh);
                sycl::accessor<float, 1, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    buffer_acc_ct1(sycl::range<1>(320 /*threads_per_block*/),
                                   cgh);
                sycl::accessor<int, 0, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    cell_converged_acc_ct1(cgh);

                auto device_IMGVF_array_ct0 = device_IMGVF_array;
                auto device_I_array_ct1 = device_I_array;
                auto device_m_array_ct2 = device_m_array;
                auto device_n_array_ct3 = device_n_array;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, num_cells) *
                            sycl::range<3>(1, 1, threads_per_block),
                        sycl::range<3>(1, 1, threads_per_block)),
                    [=](sycl::nd_item<3> item_ct1) {
                            IMGVF_kernel(device_IMGVF_array_ct0,
                                         device_I_array_ct1, device_m_array_ct2,
                                         device_n_array_ct3, (float)vx,
                                         (float)vy, (float)e, max_iterations,
                                         (float)cutoff, item_ct1,
                                         IMGVF_acc_ct1.get_pointer(),
                                         buffer_acc_ct1.get_pointer(),
                                         cell_converged_acc_ct1.get_pointer());
                    });
        });

        // Check for kernel errors
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1010:17: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        int error = 0;

        // Copy back the final results from the GPU
	IMGVF_cuda_cleanup(IMGVF, num_cells);
}


// Initializes data on the GPU for the MGVF kernel
void IMGVF_cuda_init(MAT **IE, int num_cells) {
	// Allocate arrays of pointers to device memory
	host_I_array = (float **) malloc(sizeof(float *) * num_cells);
	host_IMGVF_array = (float **) malloc(sizeof(float *) * num_cells);
        device_I_array =
            sycl::malloc_device<float *>(num_cells, dpct::get_default_queue());
        device_IMGVF_array =
            sycl::malloc_device<float *>(num_cells, dpct::get_default_queue());

        // Allocate arrays of memory dimensions
	host_m_array = (int *) malloc(sizeof(int) * num_cells);
	host_n_array = (int *) malloc(sizeof(int) * num_cells);
        device_m_array =
            sycl::malloc_device<int>(num_cells, dpct::get_default_queue());
        device_n_array =
            sycl::malloc_device<int>(num_cells, dpct::get_default_queue());

        // Figure out the size of all of the matrices combined
	int i, j, cell_num;
	int total_size = 0;
	for (cell_num = 0; cell_num < num_cells; cell_num++) {
		MAT *I = IE[cell_num];
		int size = I->m * I->n;
		total_size += size;
	}
	total_mem_size = total_size * sizeof(float);
	
	// Allocate host memory just once for all cells
	host_I_all = (float *) malloc(total_mem_size);
	
	// Allocate device memory just once for all cells
	float *device_I_all, *device_IMGVF_all;
        device_I_all = (float *)sycl::malloc_device(total_mem_size,
                                                    dpct::get_default_queue());
        device_IMGVF_all = (float *)sycl::malloc_device(
            total_mem_size, dpct::get_default_queue());

        // Copy each initial matrix into the allocated host memory
	int offset = 0;
	for (cell_num = 0; cell_num < num_cells; cell_num++) {
		MAT *I = IE[cell_num];
		
		// Determine the size of the matrix
		int m = I->m, n = I->n;
		int size = m * n;
		
		// Store memory dimensions
		host_m_array[cell_num] = m;
		host_n_array[cell_num] = n;
		
		// Store pointers to allocated memory
		float *device_I = &(device_I_all[offset]);
		float *device_IMGVF = &(device_IMGVF_all[offset]);
		host_I_array[cell_num] = device_I;
		host_IMGVF_array[cell_num] = device_IMGVF;
		
		// Copy matrix I (which is also the initial IMGVF matrix) into the overall array
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++)
				host_I_all[offset + (i * n) + j] = (float) m_get_val(I, i, j);
		
		offset += size;
	}
	
	// Copy I matrices (which are also the initial IMGVF matrices) to device
        dpct::get_default_queue()
            .memcpy(device_I_all, host_I_all, total_mem_size)
            .wait();
        dpct::get_default_queue()
            .memcpy(device_IMGVF_all, host_I_all, total_mem_size)
            .wait();

        // Copy pointer arrays to device
        dpct::get_default_queue()
            .memcpy(device_I_array, host_I_array, num_cells * sizeof(float *))
            .wait();
        dpct::get_default_queue()
            .memcpy(device_IMGVF_array, host_IMGVF_array,
                    num_cells * sizeof(float *))
            .wait();

        // Copy memory dimension arrays to device
        dpct::get_default_queue()
            .memcpy(device_m_array, host_m_array, num_cells * sizeof(int))
            .wait();
        dpct::get_default_queue()
            .memcpy(device_n_array, host_n_array, num_cells * sizeof(int))
            .wait();
}


// Copies the results of the MGVF kernel back to the host
void IMGVF_cuda_cleanup(MAT **IMGVF_out_array, int num_cells) {
	// Copy the result matrices from the device to the host
        dpct::get_default_queue()
            .memcpy(host_I_all, host_IMGVF_array[0], total_mem_size)
            .wait();

        // Copy each result matrix into its appropriate host matrix
	int cell_num, offset = 0;	
	for (cell_num = 0; cell_num < num_cells; cell_num++) {
		MAT *IMGVF_out = IMGVF_out_array[cell_num];
		
		// Determine the size of the matrix
		int m = IMGVF_out->m, n = IMGVF_out->n, i, j;
		// Pack the result into the matrix
		for (i = 0; i < m; i++)
			for (j = 0; j < n; j++)
				m_set_val(IMGVF_out, i, j, (double) host_I_all[offset + (i * n) + j]);
		
		offset += (m * n);
	}
	
	// Free device memory
        sycl::free(device_m_array, dpct::get_default_queue());
        sycl::free(device_n_array, dpct::get_default_queue());
        sycl::free(device_IMGVF_array, dpct::get_default_queue());
        sycl::free(device_I_array, dpct::get_default_queue());
        sycl::free(host_IMGVF_array[0], dpct::get_default_queue());
        sycl::free(host_I_array[0], dpct::get_default_queue());

        // Free host memory
	free(host_m_array);
	free(host_n_array);
	free(host_IMGVF_array);
	free(host_I_array);
	free(host_I_all);
}
