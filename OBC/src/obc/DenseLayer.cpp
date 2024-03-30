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
		cuda::MatrixVecMul(weights_, output_.size(), input_->size(), *input_, output_);
		cuda::VecVecAdd(output_, biases_, output_);
		return &output_;
	}
}