#pragma once

#include "Layer.h"

namespace obc {
	// Dense Layer
	// A fully connected layer.
	// Each input is connected to each output.
	// Each output has a bias.
	class DenseLayer : public Layer {
	public:
		DenseLayer(size_t input_size, size_t output_size)
			: Layer(output_size) {
			weights_.resize(input_size * output_size, 1);
			biases_.resize(output_size, 0);
		}

		const std::vector<double>* Forward(const std::vector<double>* input) override;
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override;

		const std::vector<double> Backward(const std::vector<double> output_gradients, double learning_rate) override;
		const std::vector<double> BackwardGpu(const std::vector<double> output_gradients, double learning_rate) override;

		double GetWeight(size_t input_index, size_t output_index) const {
			return weights_[input_index * biases_.size() + output_index];
		}

		LayerType GetType() const override {
			return LayerType::Dense;
		}

	private:
		std::vector<double> weights_;
		std::vector<double> biases_;
	};
}