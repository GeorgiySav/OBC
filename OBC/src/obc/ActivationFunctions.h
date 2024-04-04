#pragma once

#include "ActivationLayer.h"

namespace obc {
	class Sigmoid : public ActivationLayer {
	public:
		Sigmoid(int output_size)
			: ActivationLayer(output_size, 
				Sigmoid::sigmoid, 
				static_cast<void(*)(const std::vector<double>&, std::vector<double>&)>(cuda::ApplyFunc<cuda::FunctionType::kSigmoid>),
				Sigmoid::sigmoidPrime,
				static_cast<void(*)(const std::vector<double>&, std::vector<double>&)>(cuda::ApplyFunc<cuda::FunctionType::kSigmoidPrime>)) {}
		~Sigmoid() {}

		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kSigmoid;
			data.input_size = output_.size();
			data.output_size = output_.size();
			return data;
		}
	private:	
		static double sigmoid(double x) {
			return 1 / (1 + exp(-x));
		}
		static double sigmoidPrime(double x) {
			return sigmoid(x) * (1 - sigmoid(x));
		}
	};

	class ReLU : public ActivationLayer {
	public:
		ReLU(int output_size)
			: ActivationLayer(output_size, 
				ReLU::relu, 
				static_cast<void(*)(const std::vector<double>&, std::vector<double>&)>(cuda::ApplyFunc<cuda::FunctionType::kReLu>),
				ReLU::reluPrime,
				static_cast<void(*)(const std::vector<double>&, std::vector<double>&)>(cuda::ApplyFunc<cuda::FunctionType::kReLuPrime>)) {}
		~ReLU() {}
	
		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kReLU;
			data.input_size = output_.size();
			data.output_size = output_.size();
			return data;
		}
	private:	
		static double relu(double x) {
			return x > 0 ? x : 0;
		}
		static double reluPrime(double x) {
			return x > 0 ? 1 : 0;
		}
	};
}
