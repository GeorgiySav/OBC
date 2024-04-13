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

		const ser::LayerType GetType() const override {
			return ser::LayerType::kSigmoid;
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

		const ser::LayerType GetType() const override {
			return ser::LayerType::kReLU;
		}
	private:	
		static double relu(double x) {
			return x > 0 ? x : 0;
		}
		static double reluPrime(double x) {
			return x > 0 ? 1 : 0;
		}
	};

	class Tanh : public ActivationLayer {
	public:
		Tanh(int output_size)
			: ActivationLayer(output_size, 
								Tanh::tanh, 
								static_cast<void(*)(const std::vector<double>&, std::vector<double>&)>(cuda::ApplyFunc<cuda::FunctionType::kTanh>),
								Tanh::tanhPrime,
								static_cast<void(*)(const double*, double*, int)>(cuda::ApplyFunc<cuda::FunctionType::kTanhPrime>)) {}
		~Tanh() {}

		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kTanh;
			data.input_size = output_.size();
			data.output_size = output_.size();
			return data;
		}

		const ser::LayerType GetType() const override {
			return ser::LayerType::kTanh;
		}

	private:
		static double tanh(double x) {
			return std::tanh(x);
		}
		static double tanhPrime(double x) {
			return 1 - std::pow(std::tanh(x), 2);
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

			std::vector<double> exps(input->size());
			double sum = 0.0;
			double max = input_->at(0);
			for (int i = 1; i < input->size(); i++) {
				if (input->at(i) > max) {
					max = input->at(i);
				}
			}
			for (int i = 0; i < input->size(); i++) {
				exps[i] = std::exp(input->at(i) - max);
				sum += exps[i];
			}

			for (int i = 0; i < input->size(); i++) {
				output_[i] = exps[i] / sum;
			}

			return &output_;
		}
		const std::vector<double>* ForwardGpu(const std::vector<double>* input) override { return nullptr;}

		const std::vector<std::vector<double>> Backward(const std::vector<double> output_gradients) override {
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
			return { input_gradients };
		}

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
	
		std::vector<std::vector<double>*> GetTrainableParameters() override { {}; }

		const ser::LayerData Serialize() const override {
			ser::LayerData data;
			data.type = ser::LayerType::kSoftmax;
			data.input_size = output_.size();
			data.output_size = output_.size();
			return data;
		}

		const ser::LayerType GetType() const override {
			return ser::LayerType::kSoftmax;
		}
	private:
	};
}
