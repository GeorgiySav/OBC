#include <iostream>
#include <span>
#include <algorithm>

#include "obc/Network.h"

#include "obc/cuda utilities/cuda_ops.h"

#include "obc/example models/Xor.h"

#include "obc/example models/SerializationExample.h"

int main() {

	obc::cuda::Init();

	// obc::XorModel();

	//obc::SerializationExample();

	std::vector<double> A = {
		1, 6, 2,
		5, 3, 1,
		7, 0, 4
	};
	std::vector<double> B = {
		1, 2,
		-1, 0
	};
	std::vector<double> C = {
		0, 0,
		0, 0
	};

	obc::cuda::CrossCorrelate(
		A, 0, 3, 3,
		B, 0, 2, 2, true,
		C, 0, 2, 2,
		false
	);
	std::cout;
	

/*
0 1
2 3



*/


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