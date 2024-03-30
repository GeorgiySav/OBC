#include "Network.h"

namespace obc {
	crow::json::wvalue Network::ToJsonSpine() {
		crow::json::wvalue json;

		for (const auto& layer : layers_) {	
			switch (layer->GetType()) {
			case LayerType::Activation:
				break;
			}

		}
		return json;
	}
}