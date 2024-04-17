#pragma once

#include <chrono>

#include "../Network.h"

namespace obc {
	void TestConvoModel() {

		std::vector<std::vector<double>> inputs = {
			{
				0, 0, 1, 1, 0, 0, 
				0, 0, 1, 1, 0, 0,
				1, 1, 1, 1, 1, 1,
				1, 1, 1, 1, 1, 1,
				0, 0, 1, 1, 0, 0,
				0, 0, 1, 1, 0, 0
			},
			{
				1, 0, 0, 0, 0, 1,
				0, 1, 0, 0, 1, 0,
				0, 0, 1, 1, 0, 0,
				0, 0, 1, 1, 0, 0,
				0, 1, 0, 0, 1, 0,
				1, 0, 0, 0, 0, 1
			},
			{
				0, 0, 0, 0, 0, 0,
				0, 1, 1, 1, 1, 0,
				0, 1, 1, 1, 1, 0,
				0, 1, 1, 1, 1, 0,
				0, 1, 1, 1, 1, 0,
				0, 0, 0, 0, 0, 0
			},
			{
				1, 1, 1, 1, 1, 1,
				1, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 1,
				1, 0, 0, 0, 0, 1,
				1, 1, 1, 1, 1, 1
			}
		};
		std::vector<std::vector<double>> outputs = {
			{ 1, 0, 0, 0 },
			{ 0, 1, 0, 0 },
			{ 0, 0, 1, 0 },
			{ 0, 0, 0, 1 }
		};

		Network cpu_convo_model = {
			new MaxPoolingLayer(1, 6, 6, 2),
			new ConvolutionalLayer(1, 3, 3, 2, 1),
			new DenseLayer(4,4),
			new Softmax(4)
		};

		std::cout << "CPU Model" << std::endl;
		std::cout << "Convolutional Model created" << std::endl;

		std::cout << "Model training..." << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		// Train the model
		TrainingParameters params;
		params.learning_rate = 0.01;
		params.epochs = 10000;
		params.error = ErrorFunction::kCrossEntropy;
		cpu_convo_model.Train(inputs, outputs, params, {}, {});

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		std::cout << "Time to train: " << elapsed.count() << "s\n";

		for (int i = 0; i < inputs.size(); i++) {
			std::vector<double> prediction = cpu_convo_model.Predict(inputs[i]);
			std::cout << "Input: " << inputs[i][0] << " " << inputs[i][1]
				<< ", Output: " << prediction[0] << " " << prediction[1] << " " << prediction[2] << " " << prediction[3] << "\n"
				<< ", Expected Output: " << outputs[i][0] << " " << outputs[i][1] << " " << outputs[i][2] << " " << outputs[i][3] << " " << std::endl;
		}
		std::cout << std::endl;


	}
}
