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
	const std::vector<double>* DenseLayer::ForwardGpu(const std::vector<double>* input) {
		input_ = input;

		double* d_input = nullptr;
		double* d_output = nullptr;
		double* d_weights = nullptr;
		double* d_biases = nullptr;

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_->size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), output_.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weights), weights_.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_biases), biases_.size() * sizeof(double)));

		CUDA_CHECK(cudaMemcpy(d_input, input_->data(), input_->size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_weights, weights_.data(), weights_.size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_biases, biases_.data(), biases_.size() * sizeof(double), cudaMemcpyHostToDevice));

		cuda::MatrixVecMul(d_weights, output_.size(), input_->size(), false, d_input, d_output);
		cuda::VecVecAdd(d_output, d_biases, d_output, output_.size());

		CUDA_CHECK(cudaMemcpy(output_.data(), d_output, output_.size() * sizeof(double), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(d_input));
		CUDA_CHECK(cudaFree(d_output));
		CUDA_CHECK(cudaFree(d_weights));
		CUDA_CHECK(cudaFree(d_biases));

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

	const std::vector<double> DenseLayer::Backward(const std::vector<double> output_gradients, double learning_rate) {
		// dE/dW = dE/dY * transpose(X)	
		std::vector<double> weights_gradients(weights_.size(), 0);
		for (int j = 0; j < input_->size(); j++) {
			for (int i = 0; i < output_gradients.size(); i++) {
				int index = i * input_->size() + j;
				weights_gradients[index] = output_gradients[i] * input_->at(j);
			}	
		}

		// dE/dX = transpose(W) * dE/dY
		std::vector<double> input_gradients(input_->size(), 0);
		for (int j = 0; j < output_gradients.size(); j++) {
			for (int i = 0; i < input_->size(); i++) {
				input_gradients[i] += GetWeight(i, j) * output_gradients[j];
			}
		}

		// update weights
		for (int i = 0; i < weights_.size(); i++) {
			weights_[i] -= learning_rate * weights_gradients[i];
		}

		// dE/dB = dE/dY
		// update biases
		for (int i = 0; i < biases_.size(); i++) {
			biases_[i] -= learning_rate * output_gradients[i];
		}
		
		return input_gradients;
	}
	const std::vector<double> DenseLayer::BackwardGpu(const std::vector<double> output_gradients, double learning_rate) {
		double* d_output_gradients = nullptr;
		double* d_input = nullptr;
		double* d_weights = nullptr;
		double* d_biases = nullptr;
		double* d_weights_gradients = nullptr;
		double* d_input_gradients = nullptr;
		std::vector<double> input_gradients(input_->size(), 0);

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_gradients), output_gradients.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), input_->size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weights), weights_.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_biases), biases_.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weights_gradients), weights_.size() * sizeof(double)));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input_gradients), input_->size() * sizeof(double)));

		CUDA_CHECK(cudaMemcpy(d_output_gradients, output_gradients.data(), output_gradients.size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_input, input_->data(), input_->size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_weights, weights_.data(), weights_.size() * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_biases, biases_.data(), biases_.size() * sizeof(double), cudaMemcpyHostToDevice));

		// dE/dW = dE/dY * transpose(X)
		cuda::MatrixMatrixMul(
			output_gradients.size(), 1, input_->size(), d_output_gradients, false, d_input, false, d_weights_gradients);

		// dE/dX = transpose(W) * dE/dY
		cuda::MatrixVecMul(d_weights, input_->size(), output_.size(), false, d_output_gradients, d_input_gradients);

		// update weights
		cuda::ScalarAdd(d_weights_gradients, -learning_rate, d_weights, weights_.size());

		// dE/dB = dE/dY
		// update biases
		cuda::ScalarAdd(d_output_gradients, -learning_rate, d_biases, biases_.size());

		CUDA_CHECK(cudaMemcpy(input_gradients.data(), d_input_gradients, input_->size() * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(weights_.data(), d_weights, weights_.size() * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(biases_.data(), d_biases, biases_.size() * sizeof(double), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(d_output_gradients));
		CUDA_CHECK(cudaFree(d_input));
		CUDA_CHECK(cudaFree(d_weights));
		CUDA_CHECK(cudaFree(d_biases));
		CUDA_CHECK(cudaFree(d_weights_gradients));
		CUDA_CHECK(cudaFree(d_input_gradients));

		return input_gradients;
	}
}