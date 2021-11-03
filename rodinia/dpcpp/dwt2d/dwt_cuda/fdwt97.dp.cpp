/// 
/// @file    fdwt97.cu
/// @brief   CUDA implementation of forward 9/7 2D DWT.
/// @author  Martin Jirman (207962@mail.muni.cz)
/// @date    2011-01-20 13:18
///
///
/// Copyright (c) 2011 Martin Jirman
/// All rights reserved.
/// 
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
/// 
///     * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
///     * Redistributions in binary form must reproduce the above copyright
///       notice, this list of conditions and the following disclaimer in the
///       documentation and/or other materials provided with the distribution.
/// 
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
/// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
///

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common.h"
#include "transform_buffer.h"
#include "io.h"


namespace dwt_cuda {

 
  
  /// Wraps a buffer and methods for computing 9/7 FDWT with sliding window
  /// of specified size. Template arguments specify this size.
  /// @tparam WIN_SIZE_X  width of sliding window
  /// @tparam WIN_SIZE_Y  height of sliding window
  template <int WIN_SIZE_X, int WIN_SIZE_Y>
  class FDWT97 {
  private:
    /// Type of shared memory buffer used for 9/7 DWT.
    typedef TransformBuffer<float, WIN_SIZE_X, WIN_SIZE_Y + 7, 4> FDWT97Buffer;

    /// Actual shared buffer used for forward 9/7 DWT.
    FDWT97Buffer buffer;

    /// Difference of indices of two vertically neighboring items in buffer.
    enum { STRIDE = FDWT97Buffer::VERTICAL_STRIDE };


    /// One thread's info about loading input image
    /// @tparam CHECKED  true if loader should check for image boundaries
    template <bool CHECKED>
    struct FDWT97ColumnLoadingInfo {
      /// Loader of pixels from some input image.
      VerticalDWTPixelLoader<float, CHECKED> loader;  
      
      /// Offset of column loaded by loader. (Offset in shared buffer.)
      int offset;
    };


    /// Horizontal 9/7 FDWT on specified lines of transform buffer.
    /// @param lines      number of lines to be transformed
    /// @param firstLine  index of the first line to be transformed
    void horizontalFDWT97(const int lines, const int firstLine,
                          sycl::nd_item<3> item_ct1) {
      item_ct1.barrier();
      buffer.forEachHorizontalOdd(firstLine, lines, AddScaledSum(f97Predict1),
                                  item_ct1);
      item_ct1.barrier();
      buffer.forEachHorizontalEven(firstLine, lines, AddScaledSum(f97Update1),
                                   item_ct1);
      item_ct1.barrier();
      buffer.forEachHorizontalOdd(firstLine, lines, AddScaledSum(f97Predict2),
                                  item_ct1);
      item_ct1.barrier();
      buffer.forEachHorizontalEven(firstLine, lines, AddScaledSum(f97Update2),
                                   item_ct1);
      item_ct1.barrier();
      buffer.scaleHorizontal(scale97Div, scale97Mul, firstLine, lines,
                             item_ct1);
      item_ct1.barrier();
    }


    /// Initializes one column of shared transform buffer with 7 input pixels.
    /// Those 7 pixels will not be transformed. Also initializes given loader.
    /// @tparam CHECKED     true if loader should check for image boundaries
    /// @param column       (uninitialized) object for loading input pixels
    /// @param columnIndex  index (not offset!) of the column to be loaded
    ///                     (relative to threadblock's first column)
    /// @param input        pointer to input image in GPU memory
    /// @param sizeX        width of the input image
    /// @param sizeY        height of the input image
    /// @param firstY       index of first row to be loaded from image
    template <bool CHECKED>
    void initColumn(FDWT97ColumnLoadingInfo<CHECKED> & column,
                              const int columnIndex, const float * const input, 
                              const int sizeX, const int sizeY,
                              const int firstY, sycl::nd_item<3> item_ct1) {
      // get offset of the column with index 'columnIndex'
      column.offset = buffer.getColumnOffset(columnIndex);

      // x-coordinate of the first pixel to be loaded by given loader
      const int firstX = item_ct1.get_group(2) * WIN_SIZE_X + columnIndex;

      if (item_ct1.get_group(1) == 0) {
        // topmost block - apply mirroring rules when loading first 7 rows
        column.loader.init(sizeX, sizeY, firstX, firstY);

        // load pixels in mirrored way
        buffer[column.offset + 4 * STRIDE] = column.loader.loadFrom(input);
        buffer[column.offset + 3 * STRIDE] =
        buffer[column.offset + 5 * STRIDE] = column.loader.loadFrom(input);
        buffer[column.offset + 2 * STRIDE] =
        buffer[column.offset + 6 * STRIDE] = column.loader.loadFrom(input);
        buffer[column.offset + 1 * STRIDE] = column.loader.loadFrom(input);
        buffer[column.offset + 0 * STRIDE] = column.loader.loadFrom(input);

        // reinitialize loader to start with pixel #3 again
        column.loader.init(sizeX, sizeY, firstX, firstY + 3);
      } else {
        // non-topmost row - regular loading:
        column.loader.init(sizeX, sizeY, firstX, firstY - 4);

        // load 7 rows into the transform buffer
        for(int i = 0; i < 7; i++) {
          buffer[column.offset + i * STRIDE] = column.loader.loadFrom(input);
        }
      }
      // Now, the next pixel, which will be loaded by loader, is pixel #3.
    }


    /// Loads another WIN_SIZE_Y pixels into given column using given loader.
    /// @tparam CHECKED  true if loader should check for image boundaries
    /// @param input     input image to load from
    /// @param column    loader and offset of loaded column in shared buffer
    template <bool CHECKED>
    inline void loadWindowIntoColumn(const float * const input,
                                  FDWT97ColumnLoadingInfo<CHECKED> & column) {
      for(int i = 7; i < (7 + WIN_SIZE_Y); i++) {
        buffer[column.offset + i * STRIDE] = column.loader.loadFrom(input);
      }
    }


    /// Main GPU 9/7 FDWT entry point.
    /// @tparam CHECK_LOADS   true if boundaries should be checked when loading
    /// @tparam CHECK_WRITES  true if boundaries should be checked when writing
    /// @param in        input image
    /// @param out       output buffer
    /// @param sizeX     width of the input image 
    /// @param sizeY     height of the input image
    /// @param winSteps  number of steps of sliding window
    template <bool CHECK_LOADS, bool CHECK_WRITES>
    void transform(const float * const in, float * const out,
                              const int sizeX, const int sizeY,
                              const int winSteps, sycl::nd_item<3> item_ct1) {
      // info about columns loaded by this thread: one main column and possibly
      // one boundary column. (Only some threads load some boundary column.)
      FDWT97ColumnLoadingInfo<CHECK_LOADS> loadedColumn;
      FDWT97ColumnLoadingInfo<CHECK_LOADS> boundaryColumn;

      // Initialize first 7 lines of transform buffer.
      const int firstY = item_ct1.get_group(1) * WIN_SIZE_Y * winSteps;
      initColumn(loadedColumn, item_ct1.get_local_id(2), in, sizeX, sizeY,
                 firstY, item_ct1);

      // Some threads initialize boundary columns.
      boundaryColumn.offset = 0;
      boundaryColumn.loader.clear();
      if (item_ct1.get_local_id(2) < 7) {
        // each thread among first 7 ones gets index of one of boundary columns
        const int colId = item_ct1.get_local_id(2) +
                          ((item_ct1.get_local_id(2) < 3) ? WIN_SIZE_X : -7);

        // Thread initializes offset of the boundary column (in shared buffer),
        // first 7 pixels of the column and a loader for this column.
        initColumn(boundaryColumn, colId, in, sizeX, sizeY, firstY, item_ct1);
      }

      // horizontally transform first 7 rows in all columns
      horizontalFDWT97(7, 0, item_ct1);

      // Index of column handled by this thread. (First half of threads handle
      // even columns and others handle odd columns.)
      const int outColumnIndex = parityIdx<WIN_SIZE_X>(item_ct1);

      // writer of output linear bands - initialize it
      const int firstX = item_ct1.get_group(2) * WIN_SIZE_X + outColumnIndex;
      VerticalDWTBandWriter<float, CHECK_WRITES> writer;
      writer.init(sizeX, sizeY, firstX, firstY);

      // transform buffer offset of column transformed and saved by this thread
      const int outColumnOffset = buffer.getColumnOffset(outColumnIndex);

      // (Each iteration of this loop assumes that first 7 rows of transform 
      // buffer are already loaded with horizontally transformed coefficients.)
      for(int w = 0; w < winSteps; w++) {
        // Load another WIN_SIZE_Y lines of thread's column into the buffer.
        loadWindowIntoColumn(in, loadedColumn);

        // some threads also load boundary columns
        if (item_ct1.get_local_id(2) < 7) {
          loadWindowIntoColumn(in, boundaryColumn);
        }

        // horizontally transform all newly loaded lines
        horizontalFDWT97(WIN_SIZE_Y, 7, item_ct1);

        // Using 7 registers, remember current values of last 7 rows of
        // transform buffer. These rows are transformed horizontally only 
        // and will be used in next iteration.
        float last7Lines[7];
        for(int i = 0; i < 7; i++) {
          last7Lines[i] = buffer[outColumnOffset + (WIN_SIZE_Y + i) * STRIDE];
        }

        // vertically transform all central columns (do not scale yet)
        buffer.forEachVerticalOdd(outColumnOffset, AddScaledSum(f97Predict1));
        buffer.forEachVerticalEven(outColumnOffset, AddScaledSum(f97Update1));
        buffer.forEachVerticalOdd(outColumnOffset, AddScaledSum(f97Predict2));
        buffer.forEachVerticalEven(outColumnOffset, AddScaledSum(f97Update2));

        // Save all results of current window. Results are in transform buffer
        // at rows from #4 to #(4 + WIN_SIZE_Y). Other rows are invalid now.
        // (They only served as a boundary for vertical FDWT.)
        for(int i = 4; i < (4 + WIN_SIZE_Y); i += 2) {
          const int index = outColumnOffset + i * STRIDE;
          // Write low coefficients from column into low band ...
          writer.writeLowInto(out, buffer[index] * scale97Div);
          // ... and high coeficients into the high band.
          writer.writeHighInto(out, buffer[index + STRIDE] * scale97Mul);
        }

        // Use last 7 remembered lines as first 7 lines for next iteration.
        // As expected, these lines are already horizontally transformed.
        for(int i = 0; i < 7; i++) {
          buffer[outColumnOffset + i * STRIDE] = last7Lines[i];
        }

        // Wait for all writing threads before proceeding to loading new
        // pixels in next iteration. (Not to overwrite those which
        // are not written yet.)
        item_ct1.barrier();
      }
    }
    
    
  public:
    /// Runs one of specialized variants of 9/7 FDWT according to distance of
    /// processed pixels to image boudnary. Some variants do not check for 
    /// boudnary and thus are slightly faster.
    /// @param in     input image
    /// @param out    output buffer
    /// @param sx     width of the input image 
    /// @param sy     height of the input image
    /// @param steps  number of steps of sliding window
    static void run(const float * const input, float * const output,
                               const int sx, const int sy, const int steps,
                               sycl::nd_item<3> item_ct1,
                               FDWT97<WIN_SIZE_X, WIN_SIZE_Y> *fdwt97) {
      // object with transform buffer in shared memory

      // Compute limits of this threadblock's block of pixels and use them to
      // determine, whether this threadblock will have to deal with boundary.
      // (3 in next expressions is for radius of impulse response of 9/7 FDWT.)
      const int maxX = (item_ct1.get_group(2) + 1) * WIN_SIZE_X + 3;
      const int maxY = (item_ct1.get_group(1) + 1) * WIN_SIZE_Y * steps + 3;
      const bool atRightBoudary = maxX >= sx;
      const bool atBottomBoudary = maxY >= sy;

      // Select specialized version of code according to distance of this
      // threadblock's pixels from image boundary.
      if(atBottomBoudary) {
        // near bottom boundary => check both writing and reading
        fdwt97->transform<true, true>(input, output, sx, sy, steps, item_ct1);
      } else if(atRightBoudary) {
        // near right boundary only => check writing only
        fdwt97->transform<false, true>(input, output, sx, sy, steps, item_ct1);
      } else {
        // no nearby boundary => check nothing
        fdwt97->transform<false, false>(input, output, sx, sy, steps, item_ct1);
      }
    }
    
  }; // end of class FDWT97
  
  
    
  /// Main GPU 9/7 FDWT entry point.
  /// @param input   input image
  /// @parma output  output buffer
  /// @param sx      width of the input image 
  /// @param sy      height of the input image
  /// @param steps   number of steps of sliding window
  template <int WIN_SX, int WIN_SY>
  
  void fdwt97Kernel(const float * const input, float * const output,
                               const int sx, const int sy, const int steps,
                               sycl::nd_item<3> item_ct1,
                               FDWT97<WIN_SX, WIN_SY> *fdwt97) {
    // Excuse me, dear reader of this code - this call have to be here. If you
    // try to simply put contents of following method right here, CUDA compiler
    // (version 3.2) will spit tons of nonsense messy errors ...
    // Hope they will not break it even more in future releases.
    FDWT97<WIN_SX, WIN_SY>::run(input, output, sx, sy, steps, item_ct1, fdwt97);
  }

  
  
  /// Only computes optimal number of sliding window steps, 
  /// number of threadblocks and then lanches the 9/7 FDWT kernel.
  /// @tparam WIN_SX  width of sliding window
  /// @tparam WIN_SY  height of sliding window
  /// @param in       input image
  /// @param out      output buffer
  /// @param sx       width of the input image 
  /// @param sy       height of the input image
  template <int WIN_SX, int WIN_SY>
  void launchFDWT97Kernel (float * in, float * out, int sx, int sy) {
    // compute optimal number of steps of each sliding window
    const int steps = divRndUp(sy, 15 * WIN_SY);
    
    // prepare grid size
    sycl::range<3> gSize(1, divRndUp(sy, WIN_SY * steps), divRndUp(sx, WIN_SX));

    // run kernel, possibly measure time and finally check the call
    PERF_BEGIN
    /*
    DPCT1049:14: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor<FDWT97<WIN_SX, WIN_SY>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            fdwt97_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(gSize * sycl::range<3>(1, 1, WIN_SX),
                                           sycl::range<3>(1, 1, WIN_SX)),
                         [=](sycl::nd_item<3> item_ct1) {
                             fdwt97Kernel<WIN_SX, WIN_SY>(
                                 in, out, sx, sy, steps, item_ct1,
                                 fdwt97_acc_ct1.get_pointer());
                         });
    });
    PERF_END("        FDWT97", sx, sy)
    CudaDWTTester::checkLastKernelCall("FDWT 9/7 kernel");
  }
  
  
  
  /// Forward 9/7 2D DWT. See common rules (dwt.h) for more details.
  /// @param in      Input DWT coefficients. Should be normalized (in range 
  ///                [-0.5, 0.5]). Will not be preserved (will be overwritten).
  /// @param out     output buffer on GPU - format specified in common rules
  /// @param sizeX   width of input image (in pixels)
  /// @param sizeY   height of input image (in pixels)
  /// @param levels  number of recursive DWT levels
  void fdwt97(float * in, float * out, int sizeX, int sizeY, int levels) {
    // select right width of kernel for the size of the image
    if(sizeX >= 960) {
      launchFDWT97Kernel<192, 8>(in, out, sizeX, sizeY);
    } else if (sizeX >= 480) {
      launchFDWT97Kernel<128, 6>(in, out, sizeX, sizeY);
    } else {
      launchFDWT97Kernel<64, 6>(in, out, sizeX, sizeY);
    }
    
    // if this was not the last level, continue recursively with other levels
    if(levels > 1) {
      // copy output's LL band back into input buffer
      const int llSizeX = divRndUp(sizeX, 2);
      const int llSizeY = divRndUp(sizeY, 2);
      memCopy(in, out, llSizeX, llSizeY);
      
      // run remaining levels of FDWT
      fdwt97(in, out, llSizeX, llSizeY, levels - 1);
    }
  }
  
  

} // end of namespace dwt_cuda
