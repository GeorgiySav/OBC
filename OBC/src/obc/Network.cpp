#include "Network.h"

#include <type_traits>
#include <numeric>

namespace obc {

	// template instantiation
	template void Network::Train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, TrainingParameters t_params);
	template void Network::Train(const std::vector<const std::vector<double>*>& X, const std::vector<const std::vector<double>*>& Y, TrainingParameters t_params);

	template <typename T>
	void Network::Train(const std::vector<T>& X, const std::vector<T>& Y,
		TrainingParameters t_params) {

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

		// Train the network
		for (int epoch = 0; epoch < t_params.epochs; epoch++) {
			// shuffle the data
			std::vector<int> indicies(X.size());
			std::iota(indicies.begin(), indicies.end(), 0);
			std::shuffle(indicies.begin(), indicies.end(), std::mt19937(std::random_device()()));

			double total_error = 0.0;
			for (int b = 0; b < std::floor(X.size() / t_params.batch_size); b++) {
				std::vector<T> X_batch;
				std::vector<T> Y_batch;
				for (int i = 0; i < t_params.batch_size; i++) {
					X_batch.push_back(X[indicies[b * t_params.batch_size + i]]);
					Y_batch.push_back(Y[indicies[b * t_params.batch_size + i]]);
				}

				// perform optimization
				if (t_params.optimization == Optimization::kSGD) {
					// stochastic gradient descent
					total_error += SGD(X_batch, Y_batch,
						t_params.learning_rate, 
						error_function, error_prime);
				}
				else if (t_params.optimization == Optimization::kAdam) {
					// adam optimization
					total_error += AdamGD(X_batch, Y_batch, 
						t_params.learning_rate, t_params.beta1, t_params.beta2, t_params.epsilon, 
						error_function, error_prime);
				}
			}
			total_error /= X.size();
			std::cout << "Epoch " << epoch << ", Error: " << total_error << std::endl;
		}
	}

	struct TrainableParameter {
		std::vector<double>* parameter;
		std::vector<double> gradient;
	};
	struct AdamParameter : public TrainableParameter {
		std::vector<double> m;
		std::vector<double> v;	
		std::vector<double> m_hat;
		std::vector<double> v_hat;
	};

	template <typename T>
	double Network::SGD(const std::vector<T>& X, const std::vector<T>& Y,
		double learning_rate,
		double(*error_function)(const std::vector<double>&, const std::vector<double>&),
		std::vector<double>(*error_prime)(const std::vector<double>&, const std::vector<double>&)) {

		std::vector<TrainableParameter> trainable_parameters;
		for (int i = 0; i < layers_.size(); i++) {
			auto parameters = layers_[i]->GetTrainableParameters();
			for (int j = 0; j < parameters.size(); j++) {
				TrainableParameter p;
				p.parameter = parameters[j];
				p.gradient.resize(parameters[j]->size());
				trainable_parameters.push_back(p);
			}
		}

		double total_error = 0.0;
		for (int i = 0; i < X.size(); i++) {
			// perform forward propogation
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

			// perform backward propogation
			std::vector<double> output_gradient;
			if constexpr (std::same_as<T, std::vector<double>>)
				output_gradient = error_prime(Y[i], output);
			else if constexpr (std::same_as<T, const std::vector<double>*>)
				output_gradient = error_prime(*Y[i], output);

			int trainable_parameter_index = trainable_parameters.size() - 1;
			// assuming that when cross entropy is used, the last layer is a softmax layer
			// so we skip the last layer and the gradient is already calculated
			int last_layer = (error_function == &CrossEntropy) ? layers_.size() - 2 : layers_.size() - 1;
			for (int j = last_layer; j >= 0; j--) {

				std::vector<std::vector<double>> gradients = layers_[j]->Backward(output_gradient);
				output_gradient = gradients[0];

				for (int k = gradients.size() - 1; k >= 1; k--) {
					trainable_parameters[trainable_parameter_index].gradient = gradients[k];
					trainable_parameter_index--;
				}
			}

			// update trainable parameters
			for (const auto& p : trainable_parameters) {
				for (int j = 0; j < p.parameter->size(); j++) {
					p.parameter->at(j) -= learning_rate * (p.gradient[j] / X.size());
				}
			}
		}

		return total_error / X.size();
	}
	template<typename T>
	double Network::AdamGD(const std::vector<T>& X, const std::vector<T>& Y, 
		double learning_rate, double beta1, double beta2, double epsilon, 
		double(*error_function)(const std::vector<double>&, const std::vector<double>&),
		std::vector<double>(*error_prime)(const std::vector<double>&, const std::vector<double>&)) {

		std::vector<AdamParameter> trainable_parameters;
		for (int i = 0; i < layers_.size(); i++) {
			auto parameters = layers_[i]->GetTrainableParameters();
			for (int j = 0; j < parameters.size(); j++) {
				AdamParameter p;
				p.parameter = parameters[j];
				p.gradient.resize(parameters[j]->size());
				p.m.resize(parameters[j]->size(), 0.0);
				p.v.resize(parameters[j]->size(), 0.0);
				p.m_hat.resize(parameters[j]->size(), 0.0);
				p.v_hat.resize(parameters[j]->size(), 0.0);
				trainable_parameters.push_back(p);
			}
		}

		double total_error = 0.0;
		for (int i = 0; i < X.size(); i++) {
			// perform forward propogation
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

			// perform backward propogation
			std::vector<double> output_gradient;
			if constexpr (std::same_as<T, std::vector<double>>)
				output_gradient = error_prime(Y[i], output);
			else if constexpr (std::same_as<T, const std::vector<double>*>)
				output_gradient = error_prime(*Y[i], output);

			int trainable_parameter_index = trainable_parameters.size() - 1;
			// assuming that when cross entropy is used, the last layer is a softmax layer
			// so we skip the last layer and the gradient is already calculated
			int last_layer = (error_function == &CrossEntropy) ? layers_.size() - 2 : layers_.size() - 1;
			for (int j = last_layer; j >= 0; j--) {

				std::vector<std::vector<double>> gradients = layers_[j]->Backward(output_gradient);
				output_gradient = gradients[0];

				for (int k = gradients.size() - 1; k >= 1; k--) {
					trainable_parameters[trainable_parameter_index].gradient = gradients[k];
					trainable_parameter_index--;
				}
			}

			// update trainable parameters
			for (auto& p : trainable_parameters) {
				for (int j = 0; j < p.parameter->size(); j++) {

					// compute moving average of the gradient
					p.m[j] = beta1 * p.m[j] + (1 - beta1) * p.gradient[j];

					// compute moving average of the squared gradient
					p.v[j] = beta2 * p.v[j] + (1 - beta2) * (p.gradient[j] * p.gradient[j]);

					// bias correction
					p.m_hat[j] = p.m[j] / (1 - std::pow(beta1, i + 1));
					p.v_hat[j] = p.v[j] / (1 - std::pow(beta2, i + 1));

					// update parameter
					p.parameter->at(j) -= learning_rate * p.m_hat[j] / (std::sqrt(p.v_hat[j]) + epsilon);
				}
			}
		}

		return total_error / X.size();
	}
}