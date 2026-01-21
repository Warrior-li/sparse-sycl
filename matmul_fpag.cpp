#include<sycl/sycl.hpp>
#include<iostream>

int main(){
    sycl::queue q;
    std::cout << "Device::"
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n";
    return 0;
}