/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#pragma once

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "helper_string.h"

/*
inline void __ExitInTime(int seconds)
{
    fprintf(stdout, "> exiting in %d seconds: ", seconds);
    fflush(stdout);
    time_t t;
    int count;

    for (t=time(0)+seconds, count=seconds; time(0) < t; count--) {
        fprintf(stdout, "%d...", count);
#if defined(WIN32)
        Sleep(1000);
#else
        sleep(1);
#endif
    }

    fprintf(stdout,"done!\n\n");
    fflush(stdout);
}

#define EXIT_TIME_DELAY 2

inline void EXIT_DELAY(int return_code)
{
    __ExitInTime(EXIT_TIME_DELAY);
    exit(return_code);
}
*/

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header files, please
// refer the CUDA examples for examples of the needed CUDA headers, which may change depending
// on which CUDA functions are used.

// CUDA Runtime error messages
#ifdef __DPCT_HPP__
static const char *_cudaGetErrorEnum(int error)
{
    switch (error)
    {
        case 0:
            return "cudaSuccess";

        case 52:
            return "cudaErrorMissingConfiguration";

        case 2:
            return "cudaErrorMemoryAllocation";

        case 3:
            return "cudaErrorInitializationError";

        case 719:
            return "cudaErrorLaunchFailure";

        case 53:
            return "cudaErrorPriorLaunchFailure";

        case 702:
            return "cudaErrorLaunchTimeout";

        case 701:
            return "cudaErrorLaunchOutOfResources";

        case 98:
            return "cudaErrorInvalidDeviceFunction";

        case 9:
            return "cudaErrorInvalidConfiguration";

        case 101:
            return "cudaErrorInvalidDevice";

        case 1:
            return "cudaErrorInvalidValue";

        case 12:
            return "cudaErrorInvalidPitchValue";

        case 13:
            return "cudaErrorInvalidSymbol";

        case 205:
            return "cudaErrorMapBufferObjectFailed";

        case 206:
            return "cudaErrorUnmapBufferObjectFailed";

        case 16:
            return "cudaErrorInvalidHostPointer";

        case 17:
            return "cudaErrorInvalidDevicePointer";

        case 18:
            return "cudaErrorInvalidTexture";

        case 19:
            return "cudaErrorInvalidTextureBinding";

        case 20:
            return "cudaErrorInvalidChannelDescriptor";

        case 21:
            return "cudaErrorInvalidMemcpyDirection";

        case 22:
            return "cudaErrorAddressOfConstant";

        case 23:
            return "cudaErrorTextureFetchFailed";

        case 24:
            return "cudaErrorTextureNotBound";

        case 25:
            return "cudaErrorSynchronizationError";

        case 26:
            return "cudaErrorInvalidFilterSetting";

        case 27:
            return "cudaErrorInvalidNormSetting";

        case 28:
            return "cudaErrorMixedDeviceExecution";

        case 4:
            return "cudaErrorCudartUnloading";

        case 999:
            return "cudaErrorUnknown";

        case 31:
            return "cudaErrorNotYetImplemented";

        case 32:
            return "cudaErrorMemoryValueTooLarge";

        case 400:
            return "cudaErrorInvalidResourceHandle";

        case 600:
            return "cudaErrorNotReady";

        case 35:
            return "cudaErrorInsufficientDriver";

        case 708:
            return "cudaErrorSetOnActiveProcess";

        case 37:
            return "cudaErrorInvalidSurface";

        case 100:
            return "cudaErrorNoDevice";

        case 214:
            return "cudaErrorECCUncorrectable";

        case 302:
            return "cudaErrorSharedObjectSymbolNotFound";

        case 303:
            return "cudaErrorSharedObjectInitFailed";

        case 215:
            return "cudaErrorUnsupportedLimit";

        case 43:
            return "cudaErrorDuplicateVariableName";

        case 44:
            return "cudaErrorDuplicateTextureName";

        case 45:
            return "cudaErrorDuplicateSurfaceName";

        case 46:
            return "cudaErrorDevicesUnavailable";

        case 200:
            return "cudaErrorInvalidKernelImage";

        case 209:
            return "cudaErrorNoKernelImageForDevice";

        case 49:
            return "cudaErrorIncompatibleDriverContext";

        case 704:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case 705:
            return "cudaErrorPeerAccessNotEnabled";

        case 216:
            return "cudaErrorDeviceAlreadyInUse";

        case 5:
            return "cudaErrorProfilerDisabled";

        case 6:
            return "cudaErrorProfilerNotInitialized";

        case 7:
            return "cudaErrorProfilerAlreadyStarted";

        case 8:
            return "cudaErrorProfilerAlreadyStopped";

#if __CUDA_API_VERSION >= 0x4000

        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";
#endif

        case 127:
            return "cudaErrorStartupFailure";

        case 10000:
            return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}
#endif

#ifdef __cuda_cuda_h__
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(int error)
{
    switch (error)
    {
        case 0:
            return "CUDA_SUCCESS";

        case 1:
            return "CUDA_ERROR_INVALID_VALUE";

        case 2:
            return "CUDA_ERROR_OUT_OF_MEMORY";

        case 3:
            return "CUDA_ERROR_NOT_INITIALIZED";

        case 4:
            return "CUDA_ERROR_DEINITIALIZED";

        case 5:
            return "CUDA_ERROR_PROFILER_DISABLED";

        case 6:
            return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

        case 7:
            return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

        case 8:
            return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

        case 100:
            return "CUDA_ERROR_NO_DEVICE";

        case 101:
            return "CUDA_ERROR_INVALID_DEVICE";

        case 200:
            return "CUDA_ERROR_INVALID_IMAGE";

        case 201:
            return "CUDA_ERROR_INVALID_CONTEXT";

        case 202:
            return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

        case 205:
            return "CUDA_ERROR_MAP_FAILED";

        case 206:
            return "CUDA_ERROR_UNMAP_FAILED";

        case 207:
            return "CUDA_ERROR_ARRAY_IS_MAPPED";

        case 208:
            return "CUDA_ERROR_ALREADY_MAPPED";

        case 209:
            return "CUDA_ERROR_NO_BINARY_FOR_GPU";

        case 210:
            return "CUDA_ERROR_ALREADY_ACQUIRED";

        case 211:
            return "CUDA_ERROR_NOT_MAPPED";

        case 212:
            return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

        case 213:
            return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

        case 214:
            return "CUDA_ERROR_ECC_UNCORRECTABLE";

        case 215:
            return "CUDA_ERROR_UNSUPPORTED_LIMIT";

        case 216:
            return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

        case 300:
            return "CUDA_ERROR_INVALID_SOURCE";

        case 301:
            return "CUDA_ERROR_FILE_NOT_FOUND";

        case 302:
            return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

        case 303:
            return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

        case 304:
            return "CUDA_ERROR_OPERATING_SYSTEM";

        case 400:
            return "CUDA_ERROR_INVALID_HANDLE";

        case 500:
            return "CUDA_ERROR_NOT_FOUND";

        case 600:
            return "CUDA_ERROR_NOT_READY";

        case 719:
            return "CUDA_ERROR_LAUNCH_FAILED";

        case 701:
            return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

        case 702:
            return "CUDA_ERROR_LAUNCH_TIMEOUT";

        case 703:
            return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

        case 704:
            return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

        case 705:
            return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

        case 708:
            return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

        case 709:
            return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

        case 710:
            return "CUDA_ERROR_ASSERT";

        case 711:
            return "CUDA_ERROR_TOO_MANY_PEERS";

        case 712:
            return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

        case 713:
            return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

        case 999:
            return "CUDA_ERROR_UNKNOWN";
    }

    return "<unknown>";
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}
#endif


#ifdef CUSPARSEAPI
// cuSPARSE API errors
static const char *_cudaGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
}
#endif

#ifdef CURAND_H_
// cuRAND API errors
static const char *_cudaGetErrorEnum(int error)
{
    switch (error)
    {
        case 0:
            return "CURAND_STATUS_SUCCESS";

        case 100:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case 101:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case 102:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case 103:
            return "CURAND_STATUS_TYPE_ERROR";

        case 104:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case 105:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case 106:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case 201:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case 202:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case 203:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case 204:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case 999:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

#ifdef NV_NPPIDEFS_H
// NPP API errors
static const char *_cudaGetErrorEnum(NppStatus error)
{
    switch (error)
    {
        case NPP_NOT_SUPPORTED_MODE_ERROR:
            return "NPP_NOT_SUPPORTED_MODE_ERROR";

        case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
            return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

        case NPP_RESIZE_NO_OPERATION_ERROR:
            return "NPP_RESIZE_NO_OPERATION_ERROR";

        case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
            return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

        case NPP_BAD_ARG_ERROR:
            return "NPP_BAD_ARGUMENT_ERROR";

        case NPP_COEFF_ERROR:
            return "NPP_COEFFICIENT_ERROR";

        case NPP_RECT_ERROR:
            return "NPP_RECTANGLE_ERROR";

        case NPP_QUAD_ERROR:
            return "NPP_QUADRANGLE_ERROR";

        case NPP_MEM_ALLOC_ERR:
            return "NPP_MEMORY_ALLOCATION_ERROR";

        case NPP_HISTO_NUMBER_OF_LEVELS_ERROR:
            return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

        case NPP_INVALID_INPUT:
            return "NPP_INVALID_INPUT";

        case NPP_POINTER_ERROR:
            return "NPP_POINTER_ERROR";

        case NPP_WARNING:
            return "NPP_WARNING";

        case NPP_ODD_ROI_WARNING:
            return "NPP_ODD_ROI_WARNING";
#else

            // These are for CUDA 5.5 or higher
        case NPP_BAD_ARGUMENT_ERROR:
            return "NPP_BAD_ARGUMENT_ERROR";

        case NPP_COEFFICIENT_ERROR:
            return "NPP_COEFFICIENT_ERROR";

        case NPP_RECTANGLE_ERROR:
            return "NPP_RECTANGLE_ERROR";

        case NPP_QUADRANGLE_ERROR:
            return "NPP_QUADRANGLE_ERROR";

        case NPP_MEMORY_ALLOCATION_ERR:
            return "NPP_MEMORY_ALLOCATION_ERROR";

        case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
            return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

        case NPP_INVALID_HOST_POINTER_ERROR:
            return "NPP_INVALID_HOST_POINTER_ERROR";

        case NPP_INVALID_DEVICE_POINTER_ERROR:
            return "NPP_INVALID_DEVICE_POINTER_ERROR";
#endif

        case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
            return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

        case NPP_TEXTURE_BIND_ERROR:
            return "NPP_TEXTURE_BIND_ERROR";

        case NPP_WRONG_INTERSECTION_ROI_ERROR:
            return "NPP_WRONG_INTERSECTION_ROI_ERROR";

        case NPP_NOT_EVEN_STEP_ERROR:
            return "NPP_NOT_EVEN_STEP_ERROR";

        case NPP_INTERPOLATION_ERROR:
            return "NPP_INTERPOLATION_ERROR";

        case NPP_RESIZE_FACTOR_ERROR:
            return "NPP_RESIZE_FACTOR_ERROR";

        case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
            return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";


#if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

        case NPP_MEMFREE_ERR:
            return "NPP_MEMFREE_ERR";

        case NPP_MEMSET_ERR:
            return "NPP_MEMSET_ERR";

        case NPP_MEMCPY_ERR:
            return "NPP_MEMCPY_ERROR";

        case NPP_MIRROR_FLIP_ERR:
            return "NPP_MIRROR_FLIP_ERR";
#else

        case NPP_MEMFREE_ERROR:
            return "NPP_MEMFREE_ERROR";

        case NPP_MEMSET_ERROR:
            return "NPP_MEMSET_ERROR";

        case NPP_MEMCPY_ERROR:
            return "NPP_MEMCPY_ERROR";

        case NPP_MIRROR_FLIP_ERROR:
            return "NPP_MIRROR_FLIP_ERROR";
#endif

        case NPP_ALIGNMENT_ERROR:
            return "NPP_ALIGNMENT_ERROR";

        case NPP_STEP_ERROR:
            return "NPP_STEP_ERROR";

        case NPP_SIZE_ERROR:
            return "NPP_SIZE_ERROR";

        case NPP_NULL_POINTER_ERROR:
            return "NPP_NULL_POINTER_ERROR";

        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
            return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

        case NPP_NOT_IMPLEMENTED_ERROR:
            return "NPP_NOT_IMPLEMENTED_ERROR";

        case NPP_ERROR:
            return "NPP_ERROR";

        case NPP_SUCCESS:
            return "NPP_SUCCESS";

        case NPP_WRONG_INTERSECTION_QUAD_WARNING:
            return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

        case NPP_MISALIGNED_DST_ROI_WARNING:
            return "NPP_MISALIGNED_DST_ROI_WARNING";

        case NPP_AFFINE_QUAD_INCORRECT_WARNING:
            return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

        case NPP_DOUBLE_SIZE_WARNING:
            return "NPP_DOUBLE_SIZE_WARNING";

        case NPP_WRONG_INTERSECTION_ROI_WARNING:
            return "NPP_WRONG_INTERSECTION_ROI_WARNING";
    }

    return "<unknown>";
}
#endif

#ifdef __DPCT_HPP__
#ifndef DEVICE_RESET
#define DEVICE_RESET dpct::get_current_device().reset();
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
}

#ifdef __DPCT_HPP__
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    /*
    DPCT1010:0: SYCL uses exceptions to report errors and does not use the error
    codes. The call was replaced with 0. You need to rewrite this code.
    */
    int err = 0;
}
#endif

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct dpct_type_3a14ef
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions

#ifdef __DPCT_HPP__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID)
{
    return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_perf_device    = 0;
    int device_count       = 0, best_SM_arch      = 0;
    
    unsigned long long max_compute_perf = 0;
    dpct::device_info deviceProp;
    device_count = dpct::dev_mgr::instance().device_count();

    /*
    DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    checkCudaErrors(
        (device_count = dpct::dev_mgr::instance().device_count(), 0));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        dpct::dev_mgr::instance()
            .get_device(current_device)
            .get_device_info(deviceProp);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        /*
        DPCT1035:8: All DPC++ devices can be used by host to submit tasks. You
        may need to adjust this code.
        */
        if (true)
        {
            /*
            DPCT1005:9: The device version is different. You need to rewrite
            this code.
            */
            if (deviceProp.get_major_version() > 0 &&
                deviceProp.get_major_version() < 9999)
            {
                /*
                DPCT1005:10: The device version is different. You need to
                rewrite this code.
                */
                best_SM_arch =
                    MAX(best_SM_arch, deviceProp.get_major_version());
            }
        }

        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        dpct::dev_mgr::instance()
            .get_device(current_device)
            .get_device_info(deviceProp);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        /*
        DPCT1035:11: All DPC++ devices can be used by host to submit tasks. You
        may need to adjust this code.
        */
        if (true)
        {
            /*
            DPCT1005:12: The device version is different. You need to rewrite
            this code.
            */
            if (deviceProp.get_major_version() == 9999 &&
                deviceProp.get_minor_version() == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                /*
                DPCT1005:13: The device version is different. You need to
                rewrite this code.
                */
                sm_per_multiproc =
                    _ConvertSMVer2Cores(deviceProp.get_major_version(),
                                        deviceProp.get_minor_version());
            }

            unsigned long long compute_perf =
                (unsigned long long)deviceProp.get_max_compute_units() *
                sm_per_multiproc * deviceProp.get_max_clock_frequency();

            if (compute_perf  > max_compute_perf)
            {
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    /*
                    DPCT1005:14: The device version is different. You need to
                    rewrite this code.
                    */
                    if (deviceProp.get_major_version() == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
        }

        ++current_device;
    }

    return max_perf_device;
}


// Initialization code to find the best CUDA Device
inline int findCudaDevice(int argc, const char **argv)
{
    dpct::device_info deviceProp;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");

        if (devID < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
        else
        {
            devID = gpuDeviceInit(devID);

            if (devID < 0)
            {
                printf("exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        /*
        DPCT1003:15: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors((dpct::dev_mgr::instance().select_device(devID), 0));
        /*
        DPCT1003:16: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        checkCudaErrors(
            (dpct::dev_mgr::instance().get_device(devID).get_device_info(
                 deviceProp),
             0));
        /*
        DPCT1005:17: The device version is different. You need to rewrite this
        code.
        */
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
               deviceProp.get_name(), deviceProp.get_major_version(),
               deviceProp.get_minor_version());
    }

    return devID;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version)
{
    dpct::device_info deviceProp;
    /*
    DPCT1005:18: The device version is different. You need to rewrite this code.
    */
    deviceProp.set_major_version(0);
    /*
    DPCT1005:19: The device version is different. You need to rewrite this code.
    */
    deviceProp.set_minor_version(0);
    int dev;

    checkCudaErrors(dev = dpct::dev_mgr::instance().current_device_id());
    /*
    DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    checkCudaErrors(
        (dpct::dev_mgr::instance().get_device(dev).get_device_info(deviceProp),
         0));

    /*
    DPCT1005:21: The device version is different. You need to rewrite this code.
    */
    if ((deviceProp.get_major_version() > major_version) ||
        /*
        DPCT1005:22: The device version is different. You need to rewrite this
        code.
        */
        (deviceProp.get_major_version() == major_version &&
         deviceProp.get_minor_version() >= minor_version))
    {
        /*
        DPCT1005:23: The device version is different. You need to rewrite this
        code.
        */
        printf("  GPU Device %d: <%16s >, Compute SM %d.%d detected\n", dev,
               deviceProp.get_name(), deviceProp.get_major_version(),
               deviceProp.get_minor_version());
        return true;
    }
    else
    {
        printf("  No GPU device was found that can support CUDA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}
#endif

// end of CUDA Helper Functions


#endif
