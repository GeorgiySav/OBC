#pragma once

#include "Layer.h"

namespace obc {
	class ActivationLayer : public Layer {
	public:
		// constructor should take in an activation function as a function pointer
		ActivationLayer(size_t output_size, 
			double (*activation_function)(double), 
			void (*activation_function_gpu)(const std::vector<double>&, std::vector<double>&),
			double (*activation_derivative)(double),
			void (*activation_derivative_gpu)(const std::vector<double>&, std::vector<double>&))
			: 
			Layer(output_size), 
			activation_function_(activation_function),
			activation_function_gpu_(activation_function_gpu),
			activation_derivative_(activation_derivative),
			activation_derivative_gpu_(activation_derivative_gpu) {}
		~ActivationLayer() {}

		const std::vector<double>* Forward(const std::vector<double>* input) override {
			input_ = input;
			for (size_t i = 0; i < output_.size(); i++) {
				output_[i] = activation_function_(input_->at(i));
			}
			return &output_;
		}
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override {
			input_ = input;
			activation_function_gpu_(*input, output_);
			return &output_;
		}

		const std::vector<double> Backward(const std::vector<double> output_gradients, double learning_rate) override {
			std::vector<double> input_gradients(input_->size());
			for (size_t i = 0; i < input_gradients.size(); i++) {
				input_gradients[i] = output_gradients[i] * activation_derivative_(input_->at(i));
			}
			return input_gradients;
		}
		const std::vector<double> BackwardGpu(const std::vector<double> output_gradients, double learning_rate) override {
			std::vector<double> input_gradients(input_->size());
			activation_derivative_gpu_(output_gradients, input_gradients);
			return input_gradients;
		}

		LayerType GetType() const override {
			return LayerType::Activation;
		}
	private:
		double (*activation_function_)(double);
		void (*activation_function_gpu_)(const std::vector<double>&, std::vector<double>&);
		double (*activation_derivative_)(double);
		void (*activation_derivative_gpu_)(const std::vector<double>&, std::vector<double>&);
	};
}