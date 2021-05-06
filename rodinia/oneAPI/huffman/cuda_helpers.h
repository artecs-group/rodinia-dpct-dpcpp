#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef __CUDA_HELPERS__
#define __CUDA_HELPERS__
/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void) try {
        int count = 0;
	int i = 0;

        count = dpct::dev_mgr::instance().device_count();
        if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
                dpct::device_info prop;
                /*
		DPCT1003:12: Migrated API does not return error
                 * code. (*, 0) is inserted. You may need to rewrite this code.

                 */
                if ((dpct::dev_mgr::instance().get_device(i).get_device_info(
                         prop),
                     0) == 0) {
                        /*
			DPCT1005:13: The device version is
                         * different. You need to rewrite this code.

                         */
                        //if (prop.get_major_version() >= 1) {
                        if(dpct::get_current_device().is_gpu()){
                                break;
			}
		}
	}
	if(i == count) {
		fprintf(stdout, "There is no device supporting CUDA.\n Using first device.\n");
		//return false;
                i = 0;
	}
        dpct::dev_mgr::instance().select_device(i);

        printf("SYCL initialized.\n");
	return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif
#endif