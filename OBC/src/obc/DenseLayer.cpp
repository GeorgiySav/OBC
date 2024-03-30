#include "DenseLayer.h"

namespace obc {
	const std::vector<double>* DenseLayer::Forward(const std::vector<double>* input) {
		input_ = input;
		for (size_t i = 0; i < output_.size(); i++) {
			output_[i] = biases_[i];
			for (size_t j = 0; j < input_->size(); j++) {
				output_[i] += input_->at(j) * GetWeight(j, i);
			}
		}
		return &output_;
	}
	const std::vector<double>* DenseLayer::ForwardGpu(const std::vector<double>* input) {
		input_ = input;
		cuda::MatrixVecMul(weights_, output_.size(), input_->size(), false, *input_, output_);
		cuda::VecVecAdd(output_, biases_, output_);
		return &output_;
	}

	const std::vector<double> DenseLayer::Backward(const std::vector<double> output_gradients, double learning_rate) {
		// dE/dW = dE/dY * transpose(X)	
		std::vector<double> weights_gradients(weights_.size(), 0);
		for (size_t i = 0; i < output_gradients.size(); i++) {
			for (size_t j = 0; j < input_->size(); j++) {
				size_t index = j * input_->size() + i;
				weights_gradients[index] = output_gradients[i] * input_->at(j);
			}	
		}
		// update weights
		for (size_t i = 0; i < weights_.size(); i++) {
			weights_[i] -= learning_rate * weights_gradients[i];
		}

		// dE/dB = dE/dY
		// update biases
		for (size_t i = 0; i < biases_.size(); i++) {
			biases_[i] -= learning_rate * output_gradients[i];
		}
		
		// dE/dX = transpose(W) * dE/dY
		std::vector<double> input_gradients(input_->size(), 0);
		for (size_t j = 0; j < output_gradients.size(); j++) {
			for (size_t i = 0; i < input_->size(); i++) {
				input_gradients[i] += GetWeight(i, j) * output_gradients[j];
			}
		}
		return input_gradients;
	}
	const std::vector<double> DenseLayer::BackwardGpu(const std::vector<double> output_gradients, double learning_rate) {	
		// dE/dW = dE/dY * transpose(X)
		std::vector<double> weights_gradients(weights_.size(), 0);
		cuda::MatrixVecMul(output_gradients, 1, output_.size(), false, *input_, weights_gradients);
		// update weights
		cuda::MatrixMatrixAdd(weights_gradients, -learning_rate, weights_);

		// dE/dB = dE/dY
		// update biases
		cuda::MatrixMatrixAdd(output_gradients, -learning_rate, biases_);

		// dE/dX = transpose(W) * dE/dY
		std::vector<double> input_gradients(input_->size(), 0);
		cuda::MatrixVecMul(weights_, output_.size(), input_->size(), true, output_gradients, input_gradients);
		return input_gradients;
	}
}