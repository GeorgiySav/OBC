#include "ConvolutionalLayer.h"

namespace obc {

	void CrossCorrelate(
		const std::vector<double>& A, size_t a_offset, size_t a_height, size_t a_width,
		const std::vector<double>& B, size_t b_offset, size_t b_height, size_t b_width,
		std::vector<double>& C, size_t c_offset, size_t c_height, size_t c_width) {

		for (size_t i = 0; i <= a_width - b_width; i++) {
			for (size_t j = 0; j <= a_height - b_height; j++) {

				size_t c_index = c_offset + (j * c_width + i);

				for (size_t x = 0; x < b_width; x++) {
					for (size_t y = 0; y < b_height; y++) {

						size_t a_index = a_offset + ((j + y) * a_width + (i + x));
						size_t b_index = b_offset + (y * b_width + x);

						C[c_index] += A[a_index] * B[b_index];
					}
				}
			}
		}

	}

	void FullCrossCorrelate(
		const std::vector<double>& A, int a_offset, int a_height, int a_width,
		const std::vector<double>& B, int b_offset, int b_height, int b_width,
		std::vector<double>& C, int c_offset, int c_height, int c_width) {
	
		for (int i = 1 - b_width; i < a_width; i++) {
			for (int j = 1 - b_height; j < a_height; j++) {

				int adj_i = i + (b_width - 1);
				int adj_j = j + (b_height - 1);

				int c_index = c_offset + (adj_j * c_width + adj_i);

				for (int x = 0; x < b_width; x++) {
					for (int y = 0; y < b_height; y++) {

						if (i + x < 0 || j + y < 0
							|| i + x >= a_width || j + y >= a_height)
							continue;

						int a_index = a_offset + ((j + y) * a_width + (i + x));
						int b_index = b_offset + (y * b_width + x);

						C[c_index] += A[a_index] * B[b_index];

					}
				}

			}
		}
	
	}

	const std::vector<double>* ConvolutionalLayer::Forward(const std::vector<double>* input) {
		input_ = input;

		std::copy(biases_.begin(), biases_.end(), output_.begin());

		for (size_t s = 0; s < kernel_sets_; s++) {
			for (size_t d = 0; d < kernel_depth_; d++) {
				size_t kernel_offset = GetKernelOffset(s, d);
				size_t output_offset = GetOutputOffset(d);
				size_t input_offset = d * input_width_ * input_height_;

				CrossCorrelate(*input_, input_offset, input_height_, input_width_,
								kernels_, kernel_offset, kernel_size_, kernel_size_,
								output_, output_offset, output_height_, output_width_);
			}
		}

		return &output_;
	}

	const std::vector<double>* ConvolutionalLayer::ForwardGpu(const std::vector<double>* input) {
		return nullptr;
	}

	const std::vector<double> ConvolutionalLayer::Backward(const std::vector<double> output_gradients, double learning_rate) {

		// dE/dK_ij = X_j cross correlate dE/dY_i

		// dE/dB_i = dE/dY_I

		// dE/dX_j = sum(dE/dY_i full correlate K_ij)

		return std::vector<double>();
	}

	const std::vector<double> ConvolutionalLayer::BackwardGpu(const std::vector<double> output_gradients, double learning_rate)
	{
		return std::vector<double>();
	}

	const ser::LayerData ConvolutionalLayer::Serialize() const
	{
		return ser::LayerData();
	}

}
