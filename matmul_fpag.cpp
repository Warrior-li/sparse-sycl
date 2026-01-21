#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;

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

// 验证结果
void verifyResult(const std::vector<DataType>& C) {
    DataType sum = 0;
    for (int i = 0; i < M * N; i++) {
        sum += C[i];
    }
    std::cout << "Result checksum: " << sum << std::endl;
}

int main() {
    try {
        // 创建SYCL队列
        queue q(default_selector{});
        
        std::cout << "Running on device: " 
                  << q.get_device().get_info<info::device::name>() 
                  << std::endl;

        // 初始化矩阵
        std::vector<DataType> A(M * K);
        std::vector<DataType> B(K * N);
        std::vector<DataType> C(M * N, 0.0f);

        initMatrixA(A);
        initMatrixB(B);

        // 获取开始时间
        auto start = std::chrono::high_resolution_clock::now();

        // 创建缓冲区用于SYCL数据传输
        buffer<DataType, 1> bufA(A.data(), range<1>(M * K));
        buffer<DataType, 1> bufB(B.data(), range<1>(K * N));
        buffer<DataType, 1> bufC(C.data(), range<1>(M * N));

        // 矩阵乘法内核
        q.submit([&](handler& h) {
            auto accA = h.get_access<access::mode::read>(bufA);
            auto accB = h.get_access<access::mode::read>(bufB);
            auto accC = h.get_access<access::mode::write>(bufC);

            h.parallel_for(range<2>(M, N), [=](id<2> id) {
                int i = id[0];
                int j = id[1];
                
                DataType sum = 0.0f;
                #pragma unroll 4
                for (int k = 0; k < K; k++) {
                    sum += accA[i * K + k] * accB[k * N + j];
                }
                accC[i * N + j] = sum;
            });
        }).wait();

        // 获取结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 验证结果
        verifyResult(C);

        // 输出性能信息
        double gflops = (2.0 * M * K * N) / (duration.count() * 1e6);
        std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << std::endl;
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

        // 输出部分结果用于验证
        std::cout << "\nFirst 4x4 of result matrix C:" << std::endl;
        for (int i = 0; i < 4 && i < M; i++) {
            for (int j = 0; j < 4 && j < N; j++) {
                std::cout << C[i * N + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nMatrix multiplication completed successfully!" << std::endl;
        return 0;

    } catch (const exception& e) {
        std::cerr << "SYCL exception occurred: " << e.what() << std::endl;
        return 1;
    }
}
}