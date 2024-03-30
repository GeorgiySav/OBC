#include <iostream>

#include "obc/Network.h"

#include "obc/cuda utilities/cuda_ops.h"

int main() {

	obc::Network nn = {
		new obc::DenseLayer(3, 2),
		new obc::Sigmoid(2)
	};

	auto cpu_result = nn.Predict({ 1, 2, 3 });
	std::cout << cpu_result[0] << ", " << cpu_result[1] << std::endl;

	obc::cuda::Init();

	nn.setGpuEnabled(true);
	auto gpu_result = nn.Predict({ 1, 2, 3 });
	std::cout << gpu_result[0] << ", " << gpu_result[1] << std::endl;

	obc::cuda::Shutdown();

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