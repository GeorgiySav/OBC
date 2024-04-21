#include "DenseLayer.h"

namespace obc {
	const std::vector<double>* DenseLayer::Forward(const std::vector<double>* input) {
		input_ = input;
		for (int i = 0; i < output_.size(); i++) {
			output_[i] = biases_[i];
			for (int j = 0; j < input_->size(); j++) {
				output_[i] += input_->at(j) * GetWeight(j, i);
			}
		}
		return &output_;
	}

	const std::vector<std::vector<double>> DenseLayer::Backward(std::vector<double> output_gradients) {
		// dE/dW = dE/dY * transpose(X)	
		std::vector<double> weights_gradients(weights_.size(), 0);
		for (int i = 0; i < output_gradients.size(); i++) {
			for (int j = 0; j < input_->size(); j++) {
				int index = j * output_gradients.size() + i;
				weights_gradients[index] += output_gradients[i] * input_->at(j);
			}
		}

		// dE/dX = transpose(W) * dE/dY
		std::vector<double> input_gradients(input_->size(), 0);
		for (int j = 0; j < output_gradients.size(); j++) {
			for (int i = 0; i < input_->size(); i++) {
				input_gradients[i] += GetWeight(i, j) * output_gradients[j];
			}
		}	

		// dE/dB = dE/dY
		
		return {input_gradients, weights_gradients, output_gradients};
	}
}