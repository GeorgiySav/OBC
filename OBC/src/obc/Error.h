#pragma once

#include <vector>

namespace obc {
	enum class ErrorFunction {
		kMSE,
		kCrossEntropy
	};

	inline double Mse(const std::vector<double>& expected, const std::vector<double>& predicted) {
		double sum = 0;
		for (int i = 0; i < expected.size(); i++) {
			sum += pow(expected[i] - predicted[i], 2);
		}
		return sum / expected.size();
	}
	inline std::vector<double> MsePrime(const std::vector<double>& expected, const std::vector<double>& predicted) {
		std::vector<double> gradients;
		for (int i = 0; i < expected.size(); i++) {
			gradients.push_back(2 * (predicted[i] - expected[i]) / expected.size());
		}
		return gradients;
	}

	inline double CrossEntropy(const std::vector<double>& expected, const std::vector<double>& predicted) {
		double sum = 0;
		for (int i = 0; i < expected.size(); i++) {
			double pred = predicted[i];
			pred = std::clamp(pred, 1e-7, 1 - 1e-7);
			sum += expected[i] * std::log(pred);
		}
		return -sum;
	}

	inline std::vector<double> CrossEntropyPrime(const std::vector<double>& expected, const std::vector<double>& predicted) {	
		std::vector<double> gradients(expected.size(), 0);
		for (int i = 0; i < expected.size(); i++) {
			double pred = predicted[i];
			pred = std::clamp(pred, 1e-7, 1 - 1e-7);
			gradients[i] = pred - expected[i];
		}
		return gradients;
	}
}