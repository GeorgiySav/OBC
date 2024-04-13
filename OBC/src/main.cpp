#include <iostream>
#include <span>
#include <algorithm>

#include "obc/Network.h"

#include "obc/cuda utilities/cuda_ops.h"

#include "obc/example models/Xor.h"

#include "obc/example models/SerializationExample.h"

#include "obc/example models/TestConvoModel.h"

#include "obc/example models/MNIST/TrainMnist.h"

int main() {	

	obc::cuda::Init();

	obc::XorModel();

	//obc::SerializationExample();

	//obc::TestConvoModel();

	//obc::TrainMNIST();

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