#pragma once

#include <vector>

namespace obc {
	namespace cuda {

		extern void Init();

		extern double Dot(const std::vector<double>& A, const std::vector<double>& B);	

		extern void MatrixVecMul(const std::vector<double>& A, const size_t m, const size_t n, bool transpose, 
			const std::vector<double>& x, 
			std::vector<double>& y);

		extern void MatrixMatrixMul(
			const size_t m, const size_t k, const size_t n,
			const std::vector<double>& A, bool transposeA, 
			const std::vector<double>& B, bool transposeB, 
			std::vector<double>& C);
		
		extern void VecVecAdd(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C);

		extern void MatrixMatrixAdd(const std::vector<double>& A, double scalar, std::vector<double>& y);

		extern void VecVecElementwiseMul(const std::vector<double>& A, 
			const std::vector<double>& B, 
			std::vector<double>& C);

		enum class FunctionType {
			kSigmoid,
			kSigmoidPrime,
			kReLu,
			kReLuPrime,
		};
		template <FunctionType func>
		void ApplyFunc(std::vector<double>& A);
		template <FunctionType func>
		void ApplyFunc(const std::vector<double>& A, std::vector<double>& y);

		extern void Shutdown();
	}
}
