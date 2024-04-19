#include <iostream>
#include <span>
#include <algorithm>

#include "obc/Network.h"

#include "obc/cuda utilities/cuda_ops.h"

#include "obc/example models/Xor.h"

#include "obc/example models/SerializationExample.h"

#include "obc/example models/TestConvoModel.h"

#include "obc/example models/MNIST/TrainMnist.h"

#include <iomanip>

int main() {

	obc::ConvolutionalLayer layer(2, 3, 3, 2, 2);

	std::vector<double> inputs = {
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,

		9, 10, 11,
		12, 13, 14,
		15, 16, 17,
	};
	std::vector<double> kernels = {
		0, 1,
		2, 3,

		4, 5,
		6, 7,

		3, 2,
		1, 0,

		7, 6,
		5, 4,
	};
	std::vector<double> biases = {
		3, 4, 
		4, 3,

		4, 3, 
		4, 3,
	};
	layer.setKernels(kernels);
	layer.setBiases(biases);
	auto outputs = layer.Forward(&inputs);

	std::cout << "Outputs: " << std::endl;
	for (int i = 0; i < outputs->size(); i++) {
		std::cout << outputs->at(i) << " ";
		if ((i + 1) % 2 == 0)
			std::cout << std::endl;
	}
	

	//obc::XorModel();

	//obc::SerializationExample();

	//obc::TestConvoModel();

	obc::TrainMNIST();


	/*
	crow::SimpleApp app;

	CROW_ROUTE(app, "/")([]() {
		crow::response response{ "Hello, World!" };
		response.set_header("Access-Control-Allow-Origin", "*");
		return response;
	});

	CROW_ROUTE(app, "/echo/<string>")([](std::string message) {
		crow::json::wvalue x;
		x["message"] = message;
		crow::response response{ x };
		response.set_header("Access-Control-Allow-Origin", "*");
		return response;
	});

	CROW_ROUTE(app, "/spine")([&nn]() {
		crow::json::wvalue x = nn.ToJsonSpine();
		crow::response response{ x };
		response.set_header("Access-Control-Allow-Origin", "*");
		return response;
	});
	app.port(18080).multithreaded().run();
	*/

	return 0;
}