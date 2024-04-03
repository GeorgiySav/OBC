#pragma once

#include <random>
#include <algorithm>

#include "Layer.h"

namespace obc {

	extern void CrossCorrelate(
		const std::vector<double>& A, size_t a_offset, size_t a_height, size_t a_width,
		const std::vector<double>& B, size_t b_offset, size_t b_height, size_t b_width,
		std::vector<double>& C, size_t c_offset, size_t c_height, size_t c_width);
	extern void FullCrossCorrelate(
		const std::vector<double>& A, int a_offset, int a_height, int a_width,
		const std::vector<double>& B, int b_offset, int b_height, int b_width,
		std::vector<double>& C, int c_offset, int c_height, int c_width);

	class ConvolutionalLayer : public Layer {
	public:
		ConvolutionalLayer(
			size_t input_depth,
			size_t input_width,
			size_t input_height,
			size_t kernel_size,
			size_t output_depth)
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
	
		const ser::LayerData Serialize() const override;

//	private:
		size_t input_depth_;
		size_t input_width_;
		size_t input_height_;
		
		size_t output_depth_;
		size_t output_width_;
		size_t output_height_;

		size_t kernel_sets_;
		size_t kernel_depth_;
		size_t kernel_size_;

		std::vector<double> kernels_;
		std::vector<double> biases_;

		size_t GetKernelOffset(size_t set, size_t depth) const {
			return (set * kernel_depth_+depth) * (kernel_size_ * kernel_size_);
		}
		size_t GetKernelIndex(size_t x, size_t y) const {
			return y * kernel_size_ + x;
		}

		size_t GetOutputOffset(size_t depth) const {
			return depth * output_width_ * output_height_;
		}
		size_t GetOutputIndex(size_t x, size_t y) const {
			return y * output_width_ + x;
		}

	};
}
