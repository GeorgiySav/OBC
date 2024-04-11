#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)


namespace obc {
	namespace cuda {

		extern void Init();

		extern void Sync();

		extern void MatrixVecMul(const double* A, const int m, const int n, bool transpose, 
			const double* x, 
			double* y);
		extern void MatrixVecMul(const std::vector<double>& A, const int m, const int n, bool transpose, 
			const std::vector<double>& x, 
			std::vector<double>& y);

		extern void MatrixMatrixMul(
			const int m, const int k, const int n,
			const double* A, bool transposeA, 
			const double* B, bool transposeB, 
			double* C);
		extern void MatrixMatrixMul(
			const int m, const int k, const int n,
			const std::vector<double>& A, bool transposeA, 
			const std::vector<double>& B, bool transposeB, 
			std::vector<double>& C);
		
		extern void VecVecAdd(const double* A, const double* B, double* C, int N);
		extern void VecVecAdd(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C);

		extern void ScalarAdd(const double* A, double scalar, double* y, int N);
		extern void ScalarAdd(const std::vector<double>& A, double scalar, std::vector<double>& y);

		extern void ElementwiseMul(const double* A, const double* B, double* C, int N);
		extern void ElementwiseMul(const std::vector<double>& A, 
			const std::vector<double>& B, 
			std::vector<double>& C);

		extern void CrossCorrelate(
			const double* A, int a_offset, int a_height, int a_width,
			const double* B, int b_offset, int b_height, int b_width, bool rot180,
			double* C, int c_offset, int c_height, int c_width,
			bool full);
		extern void CrossCorrelate(
			const std::vector<double>& A, int a_offset, int a_height, int a_width,
			const std::vector<double>& B, int b_offset, int b_height, int b_width, bool rot180,
				  std::vector<double>& C, int c_offset, int c_height, int c_width,
			bool full);

		extern void Copy(const double* A, double* B, int N);
		extern void Copy(const std::vector<double>& A, std::vector<double>& B);

		enum class FunctionType {
			kSigmoid,
			kSigmoidPrime,
			kReLu,
			kReLuPrime,
			kTanh,
			kTanhPrime
		};
		template <FunctionType func>
		void ApplyFunc(double* A, int N);
		template <FunctionType func>
		void ApplyFunc(std::vector<double>& A);

		template <FunctionType func>
		void ApplyFunc(const double* A, double* y, int N);
		template <FunctionType func>
		void ApplyFunc(const std::vector<double>& A, std::vector<double>& y);

		extern void Shutdown();
	}
}
