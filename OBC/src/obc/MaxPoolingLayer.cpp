#include "MaxPoolingLayer.h"

namespace obc {
	const std::vector<double>* MaxPoolingLayer::Forward(const std::vector<double>* input) {
		input_ = input;
		for (int d = 0; d < depth_; d++) {
			int input_offset = d * input_width_ * input_height_;
			int output_offset = d * output_width_ * output_height_;

			for (int i = 0; i < output_width_; i++) {
				for (int j = 0; j < output_height_; j++) {
					int output_index = output_offset + (j * output_width_ + i);
					int max_index = -1;
					double max_value = -std::numeric_limits<double>::max();

					for (int x = 0; x < filter_size_; x++) {
						for (int y = 0; y < filter_size_; y++) {
							int input_index = input_offset + ((j * filter_size_ + y) * input_width_ + (i * filter_size_ + x));
							if (input->at(input_index) > max_value) {
								max_value = input->at(input_index);
								max_index = input_index;
							}
						}
					}
					output_[output_index] = max_value;
					max_indices_[output_index] = max_index;
				}
			}
		}
		return &output_;
	}
	const std::vector<double>* MaxPoolingLayer::ForwardGpu(const std::vector<double>* input) {
		return nullptr;
	}

	const std::vector<std::vector<double>> MaxPoolingLayer::Backward(std::vector<double> output_gradients) {
		std::vector<double> input_gradients(input_->size(), 0);
		for (int i = 0; i < output_gradients.size(); i++) {
			input_gradients[max_indices_[i]] = output_gradients[i];
		}
		return { input_gradients };
	}

	const std::vector<double> MaxPoolingLayer::Backward(const std::vector<double> output_gradients, double learning_rate) {
		std::vector<double> input_gradients(input_->size(), 0);
		for (int i = 0; i < output_gradients.size(); i++) {
			input_gradients[max_indices_[i]] = output_gradients[i];
		}
		return input_gradients;
	}
	const std::vector<double> MaxPoolingLayer::BackwardGpu(const std::vector<double> output_gradients, double learning_rate) {
		return std::vector<double>();
	}
}
