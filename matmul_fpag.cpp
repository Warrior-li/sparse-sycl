// 根据SYCL版本选择相应的头文件
// Intel oneAPI SYCL (新版):
// #include <sycl/sycl.hpp>

// 如果上面的头文件找不到，取消注释下面的行（SYCL 2020标准）:
#include <CL/sycl.hpp>

#include <iostream>
#include <vector>
#include <chrono>

// 禁用宽字符支持以避免GCC 10.2兼容性问题
#define _GLIBCXX_USE_WCHAR_T 0

// 矩阵维度
constexpr int M = 64;  // A的行数
constexpr int K = 64;  // A的列数，B的行数
constexpr int N = 64;  // B的列数

using DataType = float;

// 初始化矩阵A（M x K）
void initMatrixA(std::vector<DataType>& A) {
    for (int i = 0; i < M * K; i++) {
        A[i] = static_cast<DataType>(i % 10);
    }
}

// 初始化矩阵B（K x N）
void initMatrixB(std::vector<DataType>& B) {
    for (int i = 0; i < K * N; i++) {
        B[i] = static_cast<DataType>(i % 10);
    }
}

// 验证结果（可选）
void verifyResult(const std::vector<DataType>& C) {
    DataType sum = 0;
    for (int i = 0; i < M * N; i++) {
        sum += C[i];
    }
    std::cout << "Result checksum: " << sum << std::endl;
}

int main() {
    try {
        // 创建SYCL队列，可在FPGA上执行
        sycl::queue q(sycl::default_selector{});
        
        printf("Running on device: %s\n", 
               q.get_device().get_info<sycl::info::device::name>().c_str());

        // 初始化矩阵
        float* A = new float[M * K];
        float* B = new float[K * N];
        float* C = new float[M * N];
        
        for (int i = 0; i < M * N; i++) C[i] = 0.0f;

        initMatrixA(A, M * K);
        initMatrixB(B, K * N);

        // 创建缓冲区用于SYCL数据传输
        sycl::buffer<float, 1> bufA(A, sycl::range<1>(M * K));
        sycl::buffer<float, 1> bufB(B, sycl::range<1>(K * N));
        sycl::buffer<float, 1> bufC(C, sycl::range<1>(M * N));

        // 矩阵乘法内核
        q.submit([&](sycl::handler& h) {
            auto accA = h.get_access<sycl::access::mode::read>(bufA);
            auto accB = h.get_access<sycl::access::mode::read>(bufB);
            auto accC = h.get_access<sycl::access::mode::write>(bufC);

            h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> id) {
                int i = id[0];
                int j = id[1];
                
                float sum = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < K; k++) {
                    sum += accA[i * K + k] * accB[k * N + j];
                }
                accC[i * N + j] = sum;
            });
        }).wait();

        // 验证结果
        verifyResult(C, M * N);

        // 输出部分结果用于验证
        printf("\nFirst 4x4 of result matrix C:\n");
        for (int i = 0; i < 4 && i < M; i++) {
            for (int j = 0; j < 4 && j < N; j++) {
                printf("%f ", C[i * N + j]);
            }
            printf("\n");
        }

        printf("\nMatrix multiplication completed successfully!\n");
        
        delete[] A;
        delete[] B;
        delete[] C;
        
        return 0;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception occurred: %s\n", e.what());
        return 1;
    }
}