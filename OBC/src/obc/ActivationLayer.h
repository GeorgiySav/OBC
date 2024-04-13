#pragma once

#include <cereal/types/functional.hpp>

#include <functional>

#include "Layer.h"

namespace obc {
	class ActivationLayer : public Layer {
	public:
		// constructor should take in an activation function as a function pointer
		ActivationLayer(int output_size, 
			std::function<double(double)> activation_function,
			std::function<void(const std::vector<double>&, std::vector<double>&)> activation_function_gpu,
			std::function<double(double)> activation_derivative,
			std::function<void(const double*, double*, int)> activation_derivative_gpu)
			: 
			Layer(output_size), 
			activation_function_(activation_function),
			activation_function_gpu_(activation_function_gpu),
			activation_derivative_(activation_derivative),
			activation_derivative_gpu_(activation_derivative_gpu) {}
		~ActivationLayer() {}

		const std::vector<double>* Forward(const std::vector<double>* input) override;
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override;

		const std::vector<std::vector<double>> Backward(std::vector<double> output_gradients) override;

		const std::vector<double> Backward(const std::vector<double> output_gradients, double learning_rate) override;	
		const std::vector<double> BackwardGpu(const std::vector<double> output_gradients, double learning_rate) override;

		std::vector<std::vector<double>*> GetTrainableParameters() override {
			return {};
		}

		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			return data;
		}

	private:
		std::function<double(double)> activation_function_;
		std::function<void(const std::vector<double>&, std::vector<double>&)> activation_function_gpu_;
		std::function<double(double)> activation_derivative_;
		std::function<void(const double*, double*, int)> activation_derivative_gpu_;
	};
}