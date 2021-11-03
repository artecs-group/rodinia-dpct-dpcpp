//===============================================================================================================================================================================================================200
//	SET_DEVICE CODE
//===============================================================================================================================================================================================================200

//======================================================================================================================================================150
//	INCLUDE/DEFINE
//======================================================================================================================================================150

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "device.h" // (in library path specified to compiler)

//======================================================================================================================================================150
//	FUNCTIONS
//======================================================================================================================================================150

//====================================================================================================100
//	SET DEVICE
//====================================================================================================100

void setdevice(void){

	// variables
	int num_devices;
	int device;

	// work
        num_devices = dpct::dev_mgr::instance().device_count();
        if (num_devices > 1) {
		
		// variables
		int max_multiprocessors; 
		int max_device;
                dpct::device_info properties;

                // initialize variables
		max_multiprocessors = 0;
		max_device = 0;
		
		for (device = 0; device < num_devices; device++) {
                        dpct::dev_mgr::instance().get_device(device).get_device_info(properties);
                        if (max_multiprocessors < properties.get_max_compute_units()) {
                                max_multiprocessors = properties.get_max_compute_units();
                                max_device = device;
			}
		}
                dpct::dev_mgr::instance().select_device(max_device);
        }

}

//====================================================================================================100
//	GET LAST ERROR
//====================================================================================================100

void checkCUDAError(const char *msg)
{
        /*
	DPCT1010:6: SYCL uses exceptions to report errors and does not
         * use the error codes. The call was replaced with 0. You need to
         * rewrite this code.
	*/
        int err = 0;
        /*
	DPCT1000:5: Error handling if-stmt was detected but could not
         * be rewritten.
	*/
        if (0 != err) {
                // fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
                /*
		DPCT1009:7: SYCL uses exceptions to report errors
                 * and does not use the error codes. The original code was
                 * commented out and a warning string was inserted. You need to
                 * rewrite this code.
		*/
                printf(
                    "Cuda error: %s: %s.\n", msg, "cudaGetErrorString not supported" /*cudaGetErrorString( err)*/);
                /*
		DPCT1001:4: The statement could not be removed.

                 */
                fflush(NULL);
                exit(EXIT_FAILURE);
	}
}

//===============================================================================================================================================================================================================200
//	END SET_DEVICE CODE
//===============================================================================================================================================================================================================200
