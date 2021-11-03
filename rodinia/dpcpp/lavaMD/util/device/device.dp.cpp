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

    dpct::device_info prop;
    int dev = 0;
    int n_dev = dpct::dev_mgr::instance().device_count();

    for (int i = 0; i < n_dev; i++){ 
	dpct::dev_mgr::instance().get_device(i).get_device_info(prop);
        std::string name = prop.get_name();
	bool is_gpu = dpct::dev_mgr::instance().get_device(i).is_gpu();
	bool is_cpu = dpct::dev_mgr::instance().get_device(i).is_cpu();
#ifdef NVIDIA_GPU
    if(is_gpu && (name.find("NVIDIA") != std::string::npos)) {	
	dev = i;
	break;
    }
#elif INTEL_GPU
    if(is_gpu && (name.find("Intel(R)") != std::string::npos)) {
    	dev = i;
	break;
    }
#else
    if(is_cpu){
    	dev = i;
	break;
    }
#endif
																    }
    dpct::dev_mgr::instance().select_device(dev);
    dpct::dev_mgr::instance().get_device(dev).get_device_info(prop);
    std::cout << "Running on " << prop.get_name() << std::endl;
    
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
