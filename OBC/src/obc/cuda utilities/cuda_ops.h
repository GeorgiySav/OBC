#pragma once

#include <vector>

namespace obc {
	namespace cuda {

		extern void Init();

		extern double Dot(const std::vector<double>& A, const std::vector<double>& B);	
		extern void MatrixVecMul(const std::vector<double>& A, const size_t m, const size_t n, const std::vector<double>& x, std::vector<double>& y);
		extern void VecVecAdd(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C);

		enum class FunctionType {
			kSigmoid,
			kReLu
		};
		template <FunctionType func>
		void ApplyFunc(std::vector<double>& A);
		template <FunctionType func>
		void ApplyFunc(const std::vector<double>& A, std::vector<double>& y);

		extern void Shutdown();
	}
}
