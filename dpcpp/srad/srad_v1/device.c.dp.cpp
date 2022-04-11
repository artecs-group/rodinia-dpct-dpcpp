#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../../common.hpp"
//======================================================================================================================================================150
//	FUNCTIONS
//======================================================================================================================================================150

//====================================================================================================100
//	SET DEVICE
//====================================================================================================100

void setdevice(void){
        select_custom_device();
}

//====================================================================================================100
//	GET LAST ERROR
//====================================================================================================100

void checkCUDAError(const char *msg)
{
        /*
        DPCT1010:6: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        int err = 0;
        /*
        DPCT1000:5: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (0 != err) {
                // fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
                /*
                DPCT1009:7: SYCL uses exceptions to report errors and does not
                use the error codes. The original code was commented out and a
                warning string was inserted. You need to rewrite this code.
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