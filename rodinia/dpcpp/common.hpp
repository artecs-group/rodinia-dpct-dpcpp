#include <iostream>
#include <CL/sycl.hpp>


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
