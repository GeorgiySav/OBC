#pragma once

#include <random>
#include <algorithm>

#include "Layer.h"

namespace obc {

	extern void CrossCorrelate(
		const std::vector<double>& A, int a_offset, int a_height, int a_width,
		const std::vector<double>& B, int b_offset, int b_height, int b_width, bool rot180,
		std::vector<double>& C, int c_offset, int c_height, int c_width);
	extern void FullCrossCorrelate(
		const std::vector<double>& A, int a_offset, int a_height, int a_width,
		const std::vector<double>& B, int b_offset, int b_height, int b_width, bool rot180,
		std::vector<double>& C, int c_offset, int c_height, int c_width);

	class ConvolutionalLayer : public Layer {
	public:
		ConvolutionalLayer(
			int input_depth,
			int input_width,
			int input_height,
			int kernel_size,
			int output_depth)
			:
			Layer(output_depth * (input_width - kernel_size + 1) * (input_height - kernel_size + 1)),
			input_depth_(input_depth),
			input_width_(input_width),
			input_height_(input_height),
			output_depth_(output_depth) {

			output_width_ = input_width_ - kernel_size + 1;
			output_height_ = input_height_ - kernel_size + 1;

			kernel_sets_ = output_depth;
			kernel_depth_ = input_depth;
			kernel_size_ = kernel_size;

			std::random_device rnd_device;
			std::mt19937 engine{ rnd_device() };
			std::uniform_real_distribution<double> dist{ 0.0, 1.0 };

			kernels_.resize(kernel_sets_ * kernel_depth_ * kernel_size_ * kernel_size_);
			std::generate(kernels_.begin(), kernels_.end(), [&]() { return dist(engine); });

			biases_.resize(output_depth_ * output_width_ * output_height_);
			std::generate(biases_.begin(), biases_.end(), [&]() { return dist(engine); });
		}

		const std::vector<double>* Forward(const std::vector<double>* input) override;
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override;

		const std::vector<double> Backward(const std::vector<double> output_gradients, double learning_rate) override;
		const std::vector<double> BackwardGpu(const std::vector<double> output_gradients, double learning_rate) override;

		void setKernels(const std::vector<double>& kernels) {
			kernels_ = kernels;
		}
		void setBiases(const std::vector<double>& biases) {
			biases_ = biases;
		}

		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kConvolutional;

			data.input_size = input_depth_ * input_width_ * input_height_;
			data.output_size = output_.size();

			data.set_parameters["input_depth"] = input_depth_;
			data.set_parameters["input_width"] = input_width_;
			data.set_parameters["input_height"] = input_height_;

			data.set_parameters["kernel_size"] = kernel_size_;

			data.set_parameters["output_depth"] = output_depth_;

			data.trainable_parameters["kernels"] = kernels_;
			data.trainable_parameters["biases"] = biases_;
			return data;
		}

//	private:
		int input_depth_;
		int input_width_;
		int input_height_;
		
		int output_depth_;
		int output_width_;
		int output_height_;

		int kernel_sets_;
		int kernel_depth_;
		int kernel_size_;

		std::vector<double> kernels_;
		std::vector<double> biases_;

		int GetKernelOffset(int set, int depth) const {
			return (set * kernel_depth_+depth) * (kernel_size_ * kernel_size_);
		}
		int GetKernelIndex(int x, int y) const {
			return y * kernel_size_ + x;
		}

		int GetOutputOffset(int depth) const {
			return depth * output_width_ * output_height_;
		}
		int GetOutputIndex(int x, int y) const {
			return y * output_width_ + x;
		}

	};
}
