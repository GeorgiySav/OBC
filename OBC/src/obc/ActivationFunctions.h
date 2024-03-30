#pragma once

#include "ActivationLayer.h"

namespace obc {
	class Sigmoid : public ActivationLayer {
	public:
		Sigmoid(size_t output_size)
			: ActivationLayer(output_size, &Sigmoid::sigmoid, &cuda::ApplyFunc<cuda::FunctionType::kSigmoid>) {}
		~Sigmoid() {}
	
	private:
		static double sigmoid(double x) {
			return 1 / (1 + exp(-x));
		}
	};

	class ReLU : public ActivationLayer {
	public:
		ReLU(size_t output_size)
			: ActivationLayer(output_size, &ReLU::relu, &cuda::ApplyFunc<cuda::FunctionType::kReLu>) {}
		~ReLU() {}

	private:
		static double relu(double x) {
			return x > 0 ? x : 0;
		}
	};

}