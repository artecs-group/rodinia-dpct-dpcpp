#include <iostream>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

void select_custom_device() {
// Figure out how many devices exist
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

// NVIDIA device selector
class NvidiaGpuSelector : public sycl::device_selector {
    public:
        int operator()(const sycl::device &Device) const override {
            const std::string driver = Device.get_info<sycl::info::device::driver_version>();

            return Device.is_gpu() && (driver.find("CUDA") != std::string::npos);
        }
};

// Intel GPU selector
class IntelGpuSelector : public sycl::device_selector {
    public:
        int operator()(const sycl::device &Device) const override {
	    const std::string vendor = Device.get_info<sycl::info::device::vendor>();

            return Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos);
        }
};
