#include "ConvolutionalLayer.h"

namespace obc {

	void CrossCorrelate(
		const std::vector<double>& A, int a_offset, int a_height, int a_width,
		const std::vector<double>& B, int b_offset, int b_height, int b_width, bool rot180,
		std::vector<double>& C, int c_offset, int c_height, int c_width) {

		for (int i = 0; i <= a_width - b_width; i++) {
			for (int j = 0; j <= a_height - b_height; j++) {

				int c_index = c_offset + (j * c_width + i);

				for (int x = 0; x < b_width; x++) {
					for (int y = 0; y < b_height; y++) {

						int a_index = a_offset + ((j + y) * a_width + (i + x));
						int b_index = (y * b_width + x);
						if (rot180)
							b_index = (b_width * b_height) - 1 - b_index;
						b_index += b_offset;

						C[c_index] += A[a_index] * B[b_index];
					}
				}
			}
		}

	}

	void FullCrossCorrelate(
		const std::vector<double>& A, int a_offset, int a_height, int a_width,
		const std::vector<double>& B, int b_offset, int b_height, int b_width, bool rot180,
		std::vector<double>& C, int c_offset, int c_height, int c_width) {
	
		for (int i = 1 - b_width; i < a_width; i++) {
			for (int j = 1 - b_height; j < a_height; j++) {

				int adj_i = i + (b_width - 1);
				int adj_j = j + (b_height - 1);

				int c_index = c_offset + (adj_j * c_width + adj_i);

				for (int x = 0; x < b_width; x++) {
					if (i + x < 0 || i + x >= a_width)
						continue;
					for (int y = 0; y < b_height; y++) {

						if (j + y < 0 || j + y >= a_height)
							continue;

						int a_index = a_offset + ((j + y) * a_width + (i + x));
						int b_index = (y * b_width + x);
						if (rot180)
							b_index = (b_width * b_height) - 1 - b_index;
						b_index += b_offset;

						C[c_index] += A[a_index] * B[b_index];

					}
				}

			}
		}
	
	}

	const std::vector<double>* ConvolutionalLayer::Forward(const std::vector<double>* input) {
		input_ = input;

		std::copy(biases_.begin(), biases_.end(), output_.begin());

		for (int s = 0; s < kernel_sets_; s++) {
			for (int d = 0; d < kernel_depth_; d++) {
				int kernel_offset = GetKernelOffset(s, d);
				int output_offset = GetOutputOffset(s);
				int input_offset = d * input_width_ * input_height_;

				CrossCorrelate(*input_, input_offset, input_height_, input_width_,
								kernels_, kernel_offset, kernel_size_, kernel_size_, false,
								output_, output_offset, output_height_, output_width_);
			}
		}

		return &output_;
	}

	const std::vector<std::vector<double>> ConvolutionalLayer::Backward(std::vector<double> output_gradients) {
		// dE/dK_ij = X_j cross correlate dE/dY_i
		// dE/dB_i = dE/dY_I
		// dE/dX_j = sum(dE/dY_i full correlate K_ij)

		std::vector<double> kernel_gradients(kernel_size_ * kernel_size_ * kernel_depth_ * kernel_sets_, 0.0);
		std::vector<double> input_gradients(input_width_ * input_height_ * kernel_depth_, 0.0);

		for (int s = 0; s < kernel_sets_; s++) {
			for (int d = 0; d < kernel_depth_; d++) {
				int kernel_offset = GetKernelOffset(s, d);
				int output_offset = GetOutputOffset(s);
				int input_offset = d * input_width_ * input_height_;

				CrossCorrelate(*input_, input_offset, input_height_, input_width_,
					output_gradients, output_offset, output_height_, output_width_, false,
					kernel_gradients, kernel_offset, kernel_size_, kernel_size_);

				FullCrossCorrelate(output_gradients, output_offset, output_height_, output_width_,
					kernels_, kernel_offset, kernel_size_, kernel_size_, true,
					input_gradients, input_offset, input_height_, input_width_);
			}
		}

		return { input_gradients, kernel_gradients, output_gradients };
	}
}
