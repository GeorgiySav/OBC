#pragma once

#include "Layer.h"

namespace obc {

	class MaxPoolingLayer : public Layer {
	public:
		MaxPoolingLayer(
			int input_depth,
			int input_width,
			int input_height,
			int filter_size,
			int filter_stride
		) : Layer(std::floor((input_height - filter_size) / filter_stride + 1)
			* std::floor((input_width - filter_size) / filter_stride + 1)
			* input_depth),
			input_depth_(input_depth),
			input_width_(input_width),
			input_height_(input_height),
			filter_size_(filter_size),
			filter_stride_(filter_stride),
			output_width_(std::floor(input_width - filter_size) / filter_stride + 1),
			output_height_(std::floor(input_height - filter_size) / filter_stride + 1) {
			max_indices_.resize(output_width_ * output_height_ * input_depth_);
		}

		const std::vector<double>* Forward(const std::vector<double>* input) override {
			input_ = input;

			// slide the filter across the input
			for (int d = 0; d < input_depth_; d++) {
				int output_offset = d * output_width_ * output_height_;
				int input_offset = d * input_width_ * input_height_;

				for (int i = 0; i < output_width_; i++) {
					for (int j = 0; j < output_height_; j++) {
						double max_value = -std::numeric_limits<double>::max();
						int max_index = 0;

						for (int x = 0; x < filter_size_; x++) {
							for (int y = 0; y < filter_size_; y++) {
								int input_index = input_offset 
									+ ((j * filter_stride_ + y) * input_width_ + (i * filter_stride_ + x));
								if (input_->at(input_index) > max_value) {
									max_value = input_->at(input_index);
									max_index = input_index;
								}
							}
						}

						output_[output_offset + (j * output_width_ + i)] = max_value;
						max_indices_[output_offset + (j * output_width_ + i)] = max_index;
					}
				}	
			}

			return &output_;
		}
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override { return nullptr; }
		
		const std::vector<double> Backward(const std::vector<double> output_gradients, double learning_rate) override { 
			std::vector<double> input_gradients(input_->size(), 0.0);

			for (int i = 0; i < max_indices_.size(); i++) {
				input_gradients[max_indices_[i]] = output_gradients[i];
			}

			return input_gradients;
		}
		const std::vector<double> BackwardGpu(const std::vector<double> output_gradients, double learning_rate) override { return std::vector<double>(); }

		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kConvolutional;

			data.input_size = input_depth_ * input_width_ * input_height_;
			data.output_size = output_.size();

			data.set_parameters["input_depth"] = input_depth_;
			data.set_parameters["input_width"] = input_width_;
			data.set_parameters["input_height"] = input_height_;

			data.set_parameters["filter_size"] = filter_size_;
			data.set_parameters["filter_stride"] = filter_stride_;

			return data;
		}
	private:
		int input_depth_;
		int input_width_;
		int input_height_;

		int filter_size_;
		int filter_stride_;

		int output_width_;
		int output_height_;

		std::vector<int> max_indices_;
	};

}