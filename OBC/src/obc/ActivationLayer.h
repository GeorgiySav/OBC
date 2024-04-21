#pragma once

#include <functional>

#include "Layer.h"

namespace obc {
	class ActivationLayer : public Layer {
	public:
		// constructor should take in an activation function as a function pointer
		ActivationLayer(int output_size, 
			std::function<double(double)> activation_function,
			std::function<double(double)> activation_derivative)
			: 
			Layer(output_size), 
			activation_function_(activation_function),
			activation_derivative_(activation_derivative) {}
		~ActivationLayer() {}

		const std::vector<double>* Forward(const std::vector<double>* input) override;
		const std::vector<std::vector<double>> Backward(std::vector<double> output_gradients) override;

		std::vector<std::vector<double>*> GetTrainableParameters() override {
			return {};
		}

		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			return data;
		}

	private:
		std::function<double(double)> activation_function_;
		std::function<double(double)> activation_derivative_;
	};
}