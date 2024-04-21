#include "ActivationLayer.h"

namespace obc {

	const std::vector<double>* ActivationLayer::Forward(const std::vector<double>* input) {
		input_ = input;
		for (int i = 0; i < output_.size(); i++) {
			output_[i] = activation_function_(input_->at(i));
		}
		return &output_;
	}

	const std::vector<std::vector<double>> ActivationLayer::Backward(std::vector<double> output_gradients) {
		// dE/dx = dE/dy elementwise_mul f'(x)
		std::vector<double> input_gradients(input_->size());
		for (int i = 0; i < input_gradients.size(); i++) {
			input_gradients[i] = output_gradients[i] * activation_derivative_(input_->at(i));
		}
		return { input_gradients };
	}
}