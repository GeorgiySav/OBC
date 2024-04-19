#include "Network.h"

#include <type_traits>
#include <numeric>
#include <iomanip>

namespace obc {

	// template instantiation
	/*
	template void Network::Train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, 
								TrainingParameters t_params,
								const std::vector<std::vector<double>>& X_acc, const std::vector<int>& Y_acc);

	template void Network::Train(const std::vector<const std::vector<double>*>& X, const std::vector<const std::vector<double>*>& Y, 
								TrainingParameters t_params,
								const std::vector<const std::vector<double>*>& X_acc, const std::vector<int>& Y_acc);

	template double Network::Test(const std::vector<std::vector<double>>& X, const std::vector<int>& Y);
	*/
	template void Network::Train(
		const std::vector<std::vector<double>>&,
		const std::vector<std::vector<double>>&,
		TrainingParameters,
		const std::vector<std::vector<double>>&,
		const std::vector<int>&);
	template void Network::Train(
		const std::vector<const std::vector<double>*>&, 
		const std::vector<const std::vector<double>*>&, 
		TrainingParameters,
		const std::vector<const std::vector<double>*>&,
		const std::vector<int>&);

	template double Network::Test(const std::vector<const std::vector<double>*>& X, const std::vector<int>& Y);	
	template double Network::Test(const std::vector<std::vector<double>>& X, const std::vector<int>& Y);
	
	template <typename T>
	void Network::Train(
		const std::vector<T>& X, 
		const std::vector<T>& Y,
		TrainingParameters t_params,
		const std::vector<T>& X_val,
		const std::vector<int>& Y_val
	) {

		// set the error functions
		double (*error_function)(const std::vector<double>&, const std::vector<double>&) = nullptr;
		std::vector<double>(*error_prime)(const std::vector<double>&, const std::vector<double>&) = nullptr;
		switch (t_params.error)
		{
		case obc::ErrorFunction::kMSE:
			error_function = &Mse;
			error_prime = &MsePrime;
			break;
		case obc::ErrorFunction::kCrossEntropy:
			error_function = &CrossEntropy;
			error_prime = &CrossEntropyPrime;
			break;
		default:
			error_function = &Mse;
			error_prime = &MsePrime;
			break;
		}

		std::vector<TrainableParameter> trainable_parameters;
		for (auto& layer : layers_) {
			auto params = layer->GetTrainableParameters();
			for (auto& param : params) {
				TrainableParameter p;
				p.parameter = param;
				p.gradient.resize(param->size(), 0.0);

				if (t_params.optimiser == Optimizer::kAdam) {
					p.m.resize(param->size(), 0.0);
					p.v.resize(param->size(), 0.0);
					p.m_hat.resize(param->size(), 0.0);
					p.v_hat.resize(param->size(), 0.0);
				}

				trainable_parameters.push_back(p);
			}
		}

		// Train the network
		for (int epoch = 0; epoch < t_params.epochs; epoch++) {
			double total_error = 0.0;
			if (t_params.optimiser == Optimizer::kAdam) {
				for (auto& param : trainable_parameters) {
					std::fill(param.m.begin(), param.m.end(), 0.0);
					std::fill(param.v.begin(), param.v.end(), 0.0);
				}
			}
			for (int i = 0; i < X.size(); i++) {
				// forward propogation
				std::vector<double> output;
				if constexpr (std::same_as<T, std::vector<double>>)
					output = this->Predict(X[i]);
				else if constexpr (std::same_as<T, const std::vector<double>*>)
					output = this->Predict(*X[i]);

				// Error
				if constexpr (std::same_as<T, std::vector<double>>)
					total_error += error_function(Y[i], output);
				else if constexpr (std::same_as<T, const std::vector<double>*>)
					total_error += error_function(*Y[i], output);

				// backward propogation
				std::vector<double> output_gradient;
				if constexpr (std::same_as<T, std::vector<double>>)
					output_gradient = error_prime(Y[i], output);
				else if constexpr (std::same_as<T, const std::vector<double>*>)
					output_gradient = error_prime(*Y[i], output);

				int last_layer = layers_.size() - 1;
				if (t_params.error == ErrorFunction::kCrossEntropy) {
					// softmax layer
					last_layer--;
				}
				int trainable_index = trainable_parameters.size() - 1;
				for (int j = last_layer; j >= 0; j--) {
					auto gradients = layers_[j]->Backward(output_gradient);
					output_gradient = gradients[0];
					for (int k = gradients.size() - 1; k >= 1; k--) {
						trainable_parameters[trainable_index].gradient = gradients[k];
						trainable_index--;
					}
				}

				// update trainable parameters
				if (t_params.optimiser == Optimizer::kSGD)
					UpdateSGD(trainable_parameters, t_params.learning_rate);
				else if (t_params.optimiser == Optimizer::kAdam)
					UpdateAdam(trainable_parameters,
						t_params.learning_rate, t_params.beta1, t_params.beta2, t_params.epsilon, i);

				if (i == X.size() - 1) {
					double avg_error = total_error / i;
					double acc = 0.0;
					/*
						*/
					if (Y_val.size())
						acc = Test(X_val, Y_val);
					else
						acc = 0.0;
					std::cout << "\rEpoch " << epoch << "\t(Error: ";
					std::cout << std::fixed << std::setprecision(4) << avg_error;
					std::cout << ", Accuracy: ";
					std::cout << std::fixed << std::setprecision(2) << acc;
					std::cout << ")   Progress: " << i + 1 << "/" << X.size();
				}
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}


	template <typename T>
	double Network::Test(const std::vector<T>& X, const std::vector<int>& labels) {
		int num_correct = 0;
		for (int i = 0; i < X.size(); i++) {
			std::vector<double> output;
			if constexpr (std::same_as<T, std::vector<double>>)
				output = this->Predict(X[i]);
			else if constexpr (std::same_as<T, const std::vector<double>*>)
				output = this->Predict(*X[i]);

			// find max index
			int max_index = 0;
			for (int j = 1; j < output.size(); j++) {
				if (output[j] > output[max_index])
					max_index = j;
			}
			if (max_index == labels[i])
				num_correct++;
		}
		return (double)num_correct / X.size() * 100.0;
	}

	void Network::UpdateSGD(std::vector<TrainableParameter>& parameters, double learning_rate) {
		for (auto& param : parameters) {
			for (int i = 0; i < param.parameter->size(); i++) {
				param.parameter->at(i) -= learning_rate * param.gradient[i];
			}
		}
	}

	void Network::UpdateAdam(std::vector<TrainableParameter>& parameters, 
		double learning_rate, double beta1, double beta2, double epsilon, int t)	{
		for (auto& param : parameters) {
			for (int i = 0; i < param.parameter->size(); i++) {
				param.m[i] = beta1 * param.m[i] + (1 - beta1) * param.gradient[i];
				param.v[i] = beta2 * param.v[i] + (1 - beta2) * param.gradient[i] * param.gradient[i];

				param.m_hat[i] = param.m[i] / (1 - std::pow(beta1, t + 1));
				param.v_hat[i] = param.v[i] / (1 - std::pow(beta2, t + 1));

				param.parameter->at(i) -= learning_rate * param.m_hat[i] / (std::sqrt(param.v_hat[i]) + epsilon);
			}
		}
	}

}