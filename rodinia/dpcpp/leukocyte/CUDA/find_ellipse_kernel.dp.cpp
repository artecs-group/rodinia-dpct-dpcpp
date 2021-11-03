#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "find_ellipse_kernel.h"
// #include <cutil.h>
#include <stdio.h>


// The number of sample points in each ellipse (stencil)
#define NPOINTS 150
// The maximum radius of a sample ellipse
#define MAX_RAD 20
// The total number of sample ellipses
#define NCIRCLES 7
// The size of the structuring element used in dilation
#define STREL_SIZE (12 * 2 + 1)


// Matrix used to store the maximal GICOV score at each pixels
// Produced by the GICOV kernel and consumed by the dilation kernel
float *device_gicov;


// Constant device arrays holding the stencil parameters used by the GICOV kernel
dpct::constant_memory<float, 1> c_sin_angle(NPOINTS);
dpct::constant_memory<float, 1> c_cos_angle(NPOINTS);
dpct::constant_memory<int, 1> c_tX(NCIRCLES *NPOINTS);
dpct::constant_memory<int, 1> c_tY(NCIRCLES *NPOINTS);

// Texture references to the gradient matrices used by the GICOV kernel
/*
DPCT1059:0: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<float, 1> t_grad_x;
/*
DPCT1059:1: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<float, 1> t_grad_y;

// Kernel to find the maximal GICOV value at each pixel of a
//  video frame, based on the input x- and y-gradient matrices
void GICOV_kernel(int grad_m, float *gicov, sycl::nd_item<3> item_ct1,
                  float *c_sin_angle, float *c_cos_angle, int *c_tX, int *c_tY,
                  dpct::image_accessor_ext<float, 1> t_grad_x,
                  dpct::image_accessor_ext<float, 1> t_grad_y) {
	int i, j, k, n, x, y;
	
	// Determine this thread's pixel
        i = item_ct1.get_group(2) + MAX_RAD + 2;
        j = item_ct1.get_local_id(2) + MAX_RAD + 2;

        // Initialize the maximal GICOV score to 0
	float max_GICOV = 0.f;

	// Iterate across each stencil
	for (k = 0; k < NCIRCLES; k++) {
		// Variables used to compute the mean and variance
		//  of the gradients along the current stencil
		float sum = 0.f, M2 = 0.f, mean = 0.f;		
		
		// Iterate across each sample point in the current stencil
		for (n = 0; n < NPOINTS; n++) {
			// Determine the x- and y-coordinates of the current sample point
			y = j + c_tY[(k * NPOINTS) + n];
			x = i + c_tX[(k * NPOINTS) + n];
			
			// Compute the combined gradient value at the current sample point
			int addr = x * grad_m + y;
                        float p = t_grad_x.read(addr) * c_cos_angle[n] +
                                  t_grad_y.read(addr) * c_sin_angle[n];

                        // Update the running total
			sum += p;
			
			// Partially compute the variance
			float delta = p - mean;
			mean = mean + (delta / (float) (n + 1));
			M2 = M2 + (delta * (p - mean));
		}
		
		// Finish computing the mean
		mean = sum / ((float) NPOINTS);
		
		// Finish computing the variance
		float var = M2 / ((float) (NPOINTS - 1));
		
		// Keep track of the maximal GICOV value seen so far
		if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
	}
	
	// Store the maximal GICOV value
	gicov[(i * grad_m) + j] = max_GICOV;
}


// Sets up and invokes the GICOV kernel and returns its output
float *GICOV_CUDA(int grad_m, int grad_n, float *host_grad_x,
                  float *host_grad_y) {
   dpct::device_ext &dev_ct1 = dpct::get_current_device();
   sycl::queue &q_ct1 = dev_ct1.default_queue();

        int MaxR = MAX_RAD + 2;

	// Allocate device memory
	unsigned int grad_mem_size = sizeof(float) * grad_m * grad_n;
	float *device_grad_x, *device_grad_y;
        device_grad_x = (float *)sycl::malloc_device(grad_mem_size,
                                                     dpct::get_default_queue());
        device_grad_y = (float *)sycl::malloc_device(grad_mem_size,
                                                     dpct::get_default_queue());

        // Copy the input gradients to the device
        dpct::get_default_queue()
            .memcpy(device_grad_x, host_grad_x, grad_mem_size)
            .wait();
        dpct::get_default_queue()
            .memcpy(device_grad_y, host_grad_y, grad_mem_size)
            .wait();

        // Bind the device arrays to texture references
    t_grad_x.attach(device_grad_x, grad_mem_size);
    t_grad_y.attach(device_grad_y, grad_mem_size);

        // Allocate & initialize device memory for result
	// (some elements are not assigned values in the kernel)
        device_gicov = (float *)sycl::malloc_device(grad_mem_size,
                                                    dpct::get_default_queue());
        dpct::get_default_queue().memset(device_gicov, 0, grad_mem_size).wait();

        // Setup execution parameters
	int num_blocks = grad_n - (2 * MaxR);
	int threads_per_block = grad_m - (2 * MaxR);
    
	// Execute the GICOV kernel
        /*
        DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                c_sin_angle.init();
                c_cos_angle.init();
                c_tX.init();
                c_tY.init();

                auto c_sin_angle_ptr_ct1 = c_sin_angle.get_ptr();
                auto c_cos_angle_ptr_ct1 = c_cos_angle.get_ptr();
                auto c_tX_ptr_ct1 = c_tX.get_ptr();
                auto c_tY_ptr_ct1 = c_tY.get_ptr();

                auto t_grad_x_acc = t_grad_x.get_access(cgh);
                auto t_grad_y_acc = t_grad_y.get_access(cgh);

                auto t_grad_x_smpl = t_grad_x.get_sampler();
                auto t_grad_y_smpl = t_grad_y.get_sampler();

                auto device_gicov_ct1 = device_gicov;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, num_blocks) *
                            sycl::range<3>(1, 1, threads_per_block),
                        sycl::range<3>(1, 1, threads_per_block)),
                    [=](sycl::nd_item<3> item_ct1) {
                            GICOV_kernel(grad_m, device_gicov_ct1, item_ct1,
                                         c_sin_angle_ptr_ct1,
                                         c_cos_angle_ptr_ct1, c_tX_ptr_ct1,
                                         c_tY_ptr_ct1,
                                         dpct::image_accessor_ext<float, 1>(
                                             t_grad_x_smpl, t_grad_x_acc),
                                         dpct::image_accessor_ext<float, 1>(
                                             t_grad_y_smpl, t_grad_y_acc));
                    });
        });

        // Check for kernel errors
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1010:3: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        int error = 0;

        // Copy the result to the host
	float *host_gicov = (float *) malloc(grad_mem_size);
        dpct::get_default_queue()
            .memcpy(host_gicov, device_gicov, grad_mem_size)
            .wait();

        // Cleanup memory
        t_grad_x.detach();
        t_grad_y.detach();
        sycl::free(device_grad_x, dpct::get_default_queue());
        sycl::free(device_grad_y, dpct::get_default_queue());

        return host_gicov;
}


// Constant device array holding the structuring element used by the dilation kernel
dpct::constant_memory<float, 1> c_strel(STREL_SIZE *STREL_SIZE);

// Texture reference to the GICOV matrix used by the dilation kernel
/*
DPCT1059:5: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<float, 1> t_img;

// Kernel to compute the dilation of the GICOV matrix produced by the GICOV kernel
// Each element (i, j) of the output matrix is set equal to the maximal value in
//  the neighborhood surrounding element (i, j) in the input matrix
// Here the neighborhood is defined by the structuring element (c_strel)
void dilate_kernel(int img_m, int img_n, int strel_m, int strel_n, float *dilated,
                   sycl::nd_item<3> item_ct1, float *c_strel,
                   dpct::image_accessor_ext<float, 1> t_img) {	
	// Find the center of the structuring element
	int el_center_i = strel_m / 2;
	int el_center_j = strel_n / 2;

	// Determine this thread's location in the matrix
        int thread_id =
            (item_ct1.get_group(2) * item_ct1.get_local_range().get(2)) +
            item_ct1.get_local_id(2);
        int i = thread_id % img_m;
	int j = thread_id / img_m;

	// Initialize the maximum GICOV score seen so far to zero
	float max = 0.0;

	// Iterate across the structuring element in one dimension
	int el_i, el_j, x, y;
	for(el_i = 0; el_i < strel_m; el_i++) {
		y = i - el_center_i + el_i;
		// Make sure we have not gone off the edge of the matrix
		if( (y >= 0) && (y < img_m) ) {
			// Iterate across the structuring element in the other dimension
			for(el_j = 0; el_j < strel_n; el_j++) {
				x = j - el_center_j + el_j;
				// Make sure we have not gone off the edge of the matrix
				//  and that the current structuring element value is not zero
				if( (x >= 0) &&
					(x < img_n) &&
					(c_strel[(el_i * strel_n) + el_j] != 0) ) {
						// Determine if this is maximal value seen so far
						int addr = (x * img_m) + y;
                                                float temp = t_img.read(addr);
                                                if (temp > max) max = temp;
				}
			}
		}
	}
	
	// Store the maximum value found
	dilated[(i * img_n) + j] = max;
}


// Sets up and invokes the dilation kernel and returns its output
float *dilate_CUDA(int max_gicov_m, int max_gicov_n, int strel_m, int strel_n) {
   dpct::device_ext &dev_ct1 = dpct::get_current_device();
   sycl::queue &q_ct1 = dev_ct1.default_queue();
        // Allocate device memory for result
	unsigned int max_gicov_mem_size = sizeof(float) * max_gicov_m * max_gicov_n;
	float* device_img_dilated;
        device_img_dilated = (float *)sycl::malloc_device(
            max_gicov_mem_size, dpct::get_default_queue());

        // Bind the input matrix of GICOV values to a texture reference
        t_img.attach(device_gicov, max_gicov_mem_size);

        // Setup execution parameters
	int num_threads = max_gicov_m * max_gicov_n;
	int threads_per_block = 176;
	int num_blocks = (int) (((float) num_threads / (float) threads_per_block) + 0.5);

	// Execute the dilation kernel
        /*
        DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                c_strel.init();

                auto c_strel_ptr_ct1 = c_strel.get_ptr();

                auto t_img_acc = t_img.get_access(cgh);

                auto t_img_smpl = t_img.get_sampler();

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, 1, num_blocks) *
                            sycl::range<3>(1, 1, threads_per_block),
                        sycl::range<3>(1, 1, threads_per_block)),
                    [=](sycl::nd_item<3> item_ct1) {
                            dilate_kernel(max_gicov_m, max_gicov_n, strel_m,
                                          strel_n, device_img_dilated, item_ct1,
                                          c_strel_ptr_ct1,
                                          dpct::image_accessor_ext<float, 1>(
                                              t_img_smpl, t_img_acc));
                    });
        });

        // Check for kernel errors
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1010:7: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        int error = 0;

        // Copy the result to the host
	float *host_img_dilated = (float*) malloc(max_gicov_mem_size);
        dpct::get_default_queue()
            .memcpy(host_img_dilated, device_img_dilated, max_gicov_mem_size)
            .wait();

        // Cleanup memory
        t_img.detach();
        sycl::free(device_gicov, dpct::get_default_queue());
        sycl::free(device_img_dilated, dpct::get_default_queue());

        return host_img_dilated;
}


// Chooses the most appropriate GPU on which to execute
void select_device() {
	// Figure out how many devices exist
	int num_devices, device;
        num_devices = dpct::dev_mgr::instance().device_count();

        // Choose the device with the largest number of multiprocessors
	if (num_devices > 0) {
		int max_multiprocessors = 0, max_device = -1;
		for (device = 0; device < num_devices; device++) {
                        dpct::device_info properties;
                        dpct::dev_mgr::instance()
                            .get_device(device)
                            .get_device_info(properties);
                        if (max_multiprocessors < properties.get_max_compute_units()) {
                                max_multiprocessors = properties.get_max_compute_units();
                                max_device = device;
			}
		}
                dpct::dev_mgr::instance().select_device(max_device);
        }
	
	// The following is to remove the API initialization overhead from the runtime measurements
        sycl::free(0, dpct::get_default_queue());
}


// Transfers pre-computed constants used by the two kernels to the GPU
void transfer_constants(float *host_sin_angle, float *host_cos_angle, int *host_tX, int *host_tY, int strel_m, int strel_n, float *host_strel) {

	// Compute the sizes of the matrices
	unsigned int angle_mem_size = sizeof(float) * NPOINTS;
	unsigned int t_mem_size = sizeof(int) * NCIRCLES * NPOINTS;
	unsigned int strel_mem_size = sizeof(float) * strel_m * strel_n;

	// Copy the matrices from host memory to device constant memory
        dpct::get_default_queue()
            .memcpy(c_sin_angle.get_ptr(), host_sin_angle, angle_mem_size)
            .wait();
        dpct::get_default_queue()
            .memcpy(c_cos_angle.get_ptr(), host_cos_angle, angle_mem_size)
            .wait();
        dpct::get_default_queue().memcpy(c_tX.get_ptr(), host_tX, t_mem_size).wait();
        dpct::get_default_queue().memcpy(c_tY.get_ptr(), host_tY, t_mem_size).wait();
        dpct::get_default_queue()
            .memcpy(c_strel.get_ptr(), host_strel, strel_mem_size)
            .wait();
}
