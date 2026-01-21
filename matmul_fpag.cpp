#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>

class VectorAddId;

void VectorAdd(const int *vec_a_in, const int *vec_b_in, int *vec_c_out, int size) {
    for(int i = 0;i < size; i++) {
        vec_c_out[i] = vec_a_in[i] + vec_b_in[i];
    }
}

constexpr int kSize = 1024;

int main() {
    bool passed = true;
    try
    {
        #if FPGA_SIMULATOR
            auto selector = sycl::ext::intel::fpga_simulator_selector_v;
        #elif FPGA_HARDWARE
            auto selector = sycl::ext::intel::fpga_selector_v;
        #else
            auto selector = sycl::ext::intel::fpga_emulator_selector_v;
        #endif

        sycl::queue q(selector);

        auto device = q.get_device();


        std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}