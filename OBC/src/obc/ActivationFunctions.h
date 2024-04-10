#pragma once

#include "ActivationLayer.h"

namespace obc {
	class Sigmoid : public ActivationLayer {
	public:
		Sigmoid(int output_size)
			: ActivationLayer(output_size, 
				Sigmoid::sigmoid, 
				static_cast<void(*)(const std::vector<double>&, std::vector<double>&)>(cuda::ApplyFunc<cuda::FunctionType::kSigmoid>),
				Sigmoid::sigmoidPrime,
				static_cast<void(*)(const double*, double*, int)>(cuda::ApplyFunc<cuda::FunctionType::kSigmoidPrime>)) {}
		~Sigmoid() {}

		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kSigmoid;
			data.input_size = output_.size();
			data.output_size = output_.size();
			return data;
		}
	private:	
		static double sigmoid(double x) {
			return 1 / (1 + exp(-x));
		}
		static double sigmoidPrime(double x) {
			return sigmoid(x) * (1 - sigmoid(x));
		}
	};

	class ReLU : public ActivationLayer {
	public:
		ReLU(int output_size)
			: ActivationLayer(output_size, 
				ReLU::relu, 
				static_cast<void(*)(const std::vector<double>&, std::vector<double>&)>(cuda::ApplyFunc<cuda::FunctionType::kReLu>),
				ReLU::reluPrime,
				static_cast<void(*)(const double*, double*, int)>(cuda::ApplyFunc<cuda::FunctionType::kReLuPrime>)) {}
		~ReLU() {}
	
		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kReLU;
			data.input_size = output_.size();
			data.output_size = output_.size();
			return data;
		}
	private:	
		static double relu(double x) {
			return x > 0 ? x : 0;
		}
		static double reluPrime(double x) {
			return x > 0 ? 1 : 0;
		}
	};

	class Softmax : public Layer {
	public:
		Softmax(int output_size)
			: Layer(output_size) {}
		~Softmax() {}

		const std::vector<double>* Forward(const std::vector<double>* input) override {
			input_ = input;

			// y_i = exp(x_i) / sum(exp(x))

			for (int i = 0; i < output_.size(); i++) {
				double sum = 0.0;
				for (int j = 0; j < input_->size(); j++) {
					sum += exp(input_->at(j));
				}
				output_[i] = exp(input_->at(i)) / sum;
			}

			return &output_;
		}
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override { return nullptr;}

		const std::vector<double> Backward(const std::vector<double> output_gradients, double learning_rate) override {	
			/*
			dE/dX = (Y elementWise (I - Y)) * dE/dY
			*/
			std::vector<double> input_gradients(input_->size(), 0);
			for (int i = 0; i < input_gradients.size(); i++) {
				double sum = 0.0;
				for (int j = 0; j < output_.size(); j++) {
					sum += output_gradients[j] * output_.at(i) * ((i == j) ? (1 - output_[j]) : (-output_[j]));
				}
				input_gradients[i] = sum;
			}
			return input_gradients;
		}
		const std::vector<double> BackwardGpu(const std::vector<double> output_gradients, double learning_rate) override { return std::vector<double>(); };
		
		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kSoftmax;
			data.input_size = output_.size();
			data.output_size = output_.size();
			return data;
		}
	private:
	};
}
