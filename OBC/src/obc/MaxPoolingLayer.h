#pragma once

#include "Layer.h"

namespace obc {

	class MaxPoolingLayer : public Layer {
	public:
		MaxPoolingLayer(
			int depth,
			int input_width,
			int input_height,
			int filter_size
		) : Layer(depth * (input_width / filter_size) * (input_height / filter_size)),
			depth_(depth),
			input_width_(input_width),
			input_height_(input_height),
			filter_size_(filter_size),
			output_width_(input_width / filter_size),
			output_height_(input_height / filter_size) {
			max_indices_.resize(depth_ * output_width_ * output_height_, 0);
		}	
		~MaxPoolingLayer() {}

		const std::vector<double>* Forward(const std::vector<double>* input) override;
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override;

		const std::vector<double> Backward(const std::vector<double> output_gradients, double learning_rate) override;
		const std::vector<double> BackwardGpu(const std::vector<double> output_gradients, double learning_rate) override;

		const ser::LayerData Serialize() const {
			ser::LayerData data;
			data.type = ser::LayerType::kMaxPooling;
			data.set_parameters["depth"] = depth_;
			data.set_parameters["input_width"] = input_width_;
			data.set_parameters["input_height"] = input_height_;
			data.set_parameters["filter_size"] = filter_size_;
			return data;
		}

		const ser::LayerType GetType() const {
			return ser::LayerType::kMaxPooling;
		}

	private:
		int depth_;

		int input_width_;
		int input_height_;

		int filter_size_;

		int output_width_;
		int output_height_;

		std::vector<int> max_indices_;
	};

}