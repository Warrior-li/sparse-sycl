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
        sycl::ext::intel::fpga_simulator_selector selector;
        #elif FPGA_HARDWARE
        sycl::ext::intel::fpga_selector selector;
        #else // FPGA_EMULATOR
        sycl::ext::intel::fpga_emulator_selector selector;
        #endif

        sycl::queue q(selector);

        auto device = q.get_device();


        std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

        int *vec_a = new int[kSize];
        int *vec_b = new int[kSize];
        int *vec_c = new int[kSize];
        for(int i = 0; i < kSize; i++) {
            vec_a[i] = i;
            vec_b[i] = (kSize- i);
        }

        std::cout << "add two vectors of size " << kSize << std::endl;

        {
            sycl::buffer<int, 1> buf_a(vec_a, sycl::range<1>(kSize));
            sycl::buffer<int, 1> buf_b(vec_b, sycl::range<1>(kSize));
            sycl::buffer<int, 1> buf_c(vec_c, sycl::range<1>(kSize));

            auto e = q.submit([&](sycl::handler& h) {
                sycl::accessor accessor_a(buf_a, h, sycl::read_only);
                sycl::accessor accessor_b(buf_b, h, sycl::read_only);
                sycl::accessor accessor_c(buf_c, h, sycl::write_only, sycl::no_init);

                h.single_task<VectorAddId>([=]() {
                    VectorAdd(&accessor_a[0], &accessor_b[0], &accessor_c[0], kSize);
                });
            });

            e.wait();
        }

        for(int i = 0; i < kSize; i++) {
            int expected = vec_a[i] + vec_b[i];
            if(vec_c[i] != expected) {
                std::cout << "Mismatch at index " << i << ": "
                            << "expected " << expected << ", got " << vec_c[i] << std::endl;
                passed = false;
                break;
            }
        }

        std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

        delete[] vec_a;
        delete[] vec_b;
        delete[] vec_c;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}