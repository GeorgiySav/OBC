#pragma once

#include <chrono>

#include "../Network.h"

namespace obc {
	void TestConvoModel() {

		std::vector<std::vector<double>> inputs = {
			{
				0, 1, 0,
				1, 1, 1,
				0, 1, 0
			},
			{
				1, 0, 1,
				0, 1, 0,
				1, 0, 1
			},
			{
				0, 0, 0,
				0, 1, 0,
				0, 0, 0
			},
			{
				1, 1, 1,
				1, 0, 1,
				1, 1, 1
			}
		};
		std::vector<std::vector<double>> outputs = {
			{ 1, 0, 0, 0 },
			{ 0, 1, 0, 0 },
			{ 0, 0, 1, 0 },
			{ 0, 0, 0, 1 }
		};

		Network cpu_convo_model = {
			new ConvolutionalLayer(1, 3, 3, 2, 1),
			new ReLU(4),
			new DenseLayer(4, 10),
			new ReLU(10),
			new DenseLayer(10, 4),
			new Softmax(4)
		};

		cpu_convo_model.setGpuEnabled(false);

		std::cout << "CPU Model" << std::endl;
		std::cout << "Convolutional Model created" << std::endl;

		std::cout << "Model training..." << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		// Train the model
		cpu_convo_model.Train(inputs, outputs, 10000, 0.1, ErrorFunction::MSE);

		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> elapsed = end - start;
		std::cout << "Time to train: " << elapsed.count() << "s\n";

		for (int i = 0; i < inputs.size(); i++) {
			std::vector<double> prediction = cpu_convo_model.Predict(inputs[i]);
			std::cout << "Input: " << inputs[i][0] << " " << inputs[i][1]
				<< ", Output: " << prediction[0]
				<< ", Expected Output: " << outputs[i][0] << std::endl;
		}
		std::cout << std::endl;


	}
}
