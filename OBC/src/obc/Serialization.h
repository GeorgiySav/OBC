#pragma once

#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

namespace obc {
	namespace ser {
		enum class ArchiveType {
			Binary,
			XML,
			JSON
		};

		enum class LayerType {
			kDense,
			kSigmoid,
			kReLU,
		};

		struct LayerData {
			LayerType type;
			size_t input_size;
			size_t output_size;

			std::map<std::string, std::vector<double>> hyper_parameters;

			template<class Archive>
			void serialize(Archive& archive) {
				archive(
					CEREAL_NVP(type), 
					CEREAL_NVP(input_size),
					CEREAL_NVP(output_size),
					CEREAL_NVP(hyper_parameters));
			}
		};
	}
}
