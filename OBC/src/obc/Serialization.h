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
			kConvolutional,
			kMaxPooling,
			kSigmoid,
			kReLU,
			kSoftmax,
		};

		struct LayerData {
			LayerType type;
			int input_size;
			int output_size;

			std::map<std::string, int> set_parameters;
			std::map<std::string, std::vector<double>> trainable_parameters;

			template<class Archive>
			void serialize(Archive& archive) {
				archive(
					CEREAL_NVP(type), 
					CEREAL_NVP(input_size),
					CEREAL_NVP(output_size),
					CEREAL_NVP(set_parameters),
					CEREAL_NVP(trainable_parameters));
			}
		};
	}
}
