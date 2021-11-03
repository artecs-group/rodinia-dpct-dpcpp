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
        dpct::device_info prop;
        int dev = 0;
        int n_dev = cl::sycl::device::get_devices(cl::sycl::info::device_type::all).size();

        if(n_dev == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

        for (int i = 0; i < n_dev; i++) {
                dpct::dev_mgr::instance().get_device(i).get_device_info(prop);
                std::string name = prop.get_name();
                bool is_gpu = dpct::dev_mgr::instance().get_device(i).is_gpu();
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
#endif
    }
        dpct::dev_mgr::instance().select_device(dev);
        dpct::dev_mgr::instance().get_device(dev).get_device_info(prop);
        std::cout << "Running on " << prop.get_name() << std::endl;
	return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif
#endif