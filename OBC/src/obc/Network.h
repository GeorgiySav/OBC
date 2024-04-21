#pragma once

#include <memory>
#include <vector>
#include <fstream>

#include "Layer.h"
#include "DenseLayer.h"
#include "ActivationFunctions.h"
#include "ConvolutionalLayer.h"
#include "MaxPoolingLayer.h"
#include "Error.h"

namespace obc {

	enum class Optimizer {
		kSGD,
		kAdam
	};

	struct TrainingParameters {
		double learning_rate = 0.1;
		int epochs = 10;
		ErrorFunction error = ErrorFunction::kMSE;
		Optimizer optimiser = Optimizer::kSGD;

		double beta1 = 0.9;
		double beta2 = 0.999;
		double epsilon = 1e-8;
	};

	class Network {
	public:
		Network() {}
		Network(std::initializer_list<Layer*> layers) {
			for (auto layer : layers) {
				layers_.push_back(std::unique_ptr<Layer>(layer));
			}
		}
		~Network() {	
		}

		std::vector<double> Predict(const std::vector<double>& input) {
			const std::vector<double>* output = &input;
			for (auto& layer : layers_) {
				output = layer->Forward(output);
			}
			return *output;
		}

		// gradient descent based training
		template <typename T>
		void Train(
			const std::vector<T>& X, 
			const std::vector<T>& Y, 
			TrainingParameters t_params,
			const std::vector<T>& X_val,
			const std::vector<int>& Y_val);

		template <typename T>
		double Test(const std::vector<T>& X, const std::vector<int>& labels);

		void Serialize(const std::string& file_name, ser::ArchiveType type) const {
			std::ofstream file(file_name, std::ios::binary);
			if (!file.is_open()) {
				throw std::runtime_error("Could not open file for writing");
			}

			std::vector<ser::LayerData> data = GetLayerData();
			if (type == ser::ArchiveType::Binary){
				cereal::BinaryOutputArchive oarchive(file); // Create an output archive
				oarchive(data); // Write the data to the archive
			}
			else if (type == ser::ArchiveType::XML) {
				cereal::XMLOutputArchive oarchive(file); // Create an output archive
				oarchive(data); // Write the data to the archive
			}
			else if (type == ser::ArchiveType::JSON) {
				cereal::JSONOutputArchive oarchive(file); // Create an output archive
				oarchive(data); // Write the data to the archive
			}	
		}
		void Deserialize(const std::string& file_name, ser::ArchiveType type) {
			std::ifstream file(file_name, std::ios::binary);
			if (!file.is_open()) {
				throw std::runtime_error("Could not open file for reading");
			}

			std::vector<ser::LayerData> data;
			if (type == ser::ArchiveType::Binary) {
				cereal::BinaryInputArchive iarchive(file); // Create an input archive
				iarchive(data); // Read the data from the archive
			}
			else if (type == ser::ArchiveType::XML) {
				cereal::XMLInputArchive iarchive(file); // Create an input archive
				iarchive(data); // Read the data from the archive
			}
			else if (type == ser::ArchiveType::JSON) {
				cereal::JSONInputArchive iarchive(file); // Create an input archive
				iarchive(data); // Read the data from the archive
			}

			LoadLayerData(data);
		}

	private:
		struct TrainableParameter {
			std::vector<double>* parameter;
			std::vector<double> gradient;

			std::vector<double> m;
			std::vector<double> v;
			std::vector<double> m_hat;
			std::vector<double> v_hat;
		};

		void UpdateSGD(std::vector<TrainableParameter>& parameters, double learning_rate);
		void UpdateAdam(std::vector<TrainableParameter>& parameters, double learning_rate, double beta1, double beta2, double epsilon, int t);

		std::vector<ser::LayerData> GetLayerData() const {
			std::vector<ser::LayerData> data;
			for (const auto& layer : layers_) {
				data.push_back(layer->Serialize());
			}
			return data;
		}

		void LoadLayerData(const std::vector<ser::LayerData>& data) {
			layers_.clear();
			for (int i = 0; i < data.size(); i++) {
				switch (data[i].type) {
				case ser::LayerType::kDense: {
					DenseLayer* new_layer = new DenseLayer(data[i].input_size, data[i].output_size);
					new_layer->SetWeights(data[i].trainable_parameters.at("weights"));
					new_layer->SetBiases(data[i].trainable_parameters.at("biases"));
					layers_.push_back(std::unique_ptr<Layer>(new_layer));
				}
					break;
				case ser::LayerType::kConvolutional: {
					ConvolutionalLayer* new_layer = new ConvolutionalLayer(
						data[i].set_parameters.at("input_depth"),
						data[i].set_parameters.at("input_width"),
						data[i].set_parameters.at("input_height"),
						data[i].set_parameters.at("kernel_size"),
						data[i].set_parameters.at("output_depth"));
					new_layer->setKernels(data[i].trainable_parameters.at("kernels"));
					new_layer->setBiases(data[i].trainable_parameters.at("biases"));
					layers_.push_back(std::unique_ptr<Layer>(new_layer));
				}
					break;
				case ser::LayerType::kMaxPooling: {
					MaxPoolingLayer* new_layer = new MaxPoolingLayer(
						data[i].set_parameters.at("depth"),
						data[i].set_parameters.at("input_width"),
						data[i].set_parameters.at("input_height"),
						data[i].set_parameters.at("filter_size"));
					layers_.push_back(std::unique_ptr<Layer>(new_layer));
				}
					break;
				case ser::LayerType::kSigmoid:
					layers_.push_back(std::make_unique<Sigmoid>(data[i].output_size));
					break;
				case ser::LayerType::kReLU:
					layers_.push_back(std::make_unique<ReLU>(data[i].output_size));
					break;
				case ser::LayerType::kSoftmax:
					layers_.push_back(std::make_unique<Softmax>(data[i].output_size));
					break;
				default:
					throw std::runtime_error("Unknown layer type");
				}
			}
		}

		std::vector<std::unique_ptr<Layer>> layers_;
	};
}