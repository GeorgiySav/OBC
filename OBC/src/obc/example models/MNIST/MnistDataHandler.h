#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <random>

namespace obc {

	class MnistData {
	public:
		MnistData() {
			image_data_ = std::vector<double>();
		}
		~MnistData() {
		}

		void TranslateImage(int x_offset, int y_offset) {
			if (x_offset < 0) {
				for (int i = 0; i < 28 + x_offset; i++) {
					for (int j = 0; j < 28; j++) {
						image_data_[j * 28 + i] = image_data_[j * 28 + i - x_offset];
					}
				}
			}
			else if (x_offset > 0) {
				for (int i = 27; i >= x_offset; i--) {
					for (int j = 0; j < 28; j++) {
						image_data_[j * 28 + i] = image_data_[j * 28 + i - x_offset];
					}
				}
			}

			if (y_offset < 0) {
				for (int i = 0; i < 28 + y_offset; i++) {
					for (int j = 0; j < 28; j++) {
						image_data_[i * 28 + j] = image_data_[(i - y_offset) * 28 + j];
					}
				}
			}
			else if (y_offset > 0) {
				for (int i = 27; i >= y_offset; i--) {
					for (int j = 0; j < 28; j++) {
						image_data_[i * 28 + j] = image_data_[(i - y_offset) * 28 + j];
					}
				}
			}
		}

		// increase the size of the image within the 28x28 grid
		void ScaleImage(double factor) {
			std::vector<double> new_image(28 * 28, 0.0);
			for (int y = 0; y < 28; y++) {
				for (int x = 0; x < 28; x++) {
					int prev_x = (int)(x / factor);
					int prev_y = (int)(y / factor);
					if (prev_x >= 28 || prev_y >= 28 || prev_x < 0 || prev_y < 0) continue;
					new_image[y * 28 + x] = image_data_[prev_y * 28 + prev_x];
				}
			}
			image_data_ = new_image;
		}

		void ApplyNoise(double noise_factor) {
			std::random_device rnd_device;
			std::mt19937 engine{ rnd_device() };
			std::normal_distribution<double> dist{ 0, noise_factor };

			for (int i = 0; i < image_data_.size(); i++) {
				image_data_[i] += dist(engine);
				if (image_data_[i] < 0.0) image_data_[i] = 0.0;
				if (image_data_[i] > 1.0) image_data_[i] = 1.0;
			}
		}

		void SetImageData(const std::vector<double>& feature_vector) {
			this->image_data_ = feature_vector;
		}
		void AppendImageData(double feature) {
			this->image_data_.push_back(feature);
		}
		void SetLabel(uint8_t label) {
			this->label_ = label;
		}	

		int GetImageSize() {
			return this->image_data_.size();
		}
		const std::vector<double>& GetImageData() {
			return this->image_data_;
		}
		uint8_t GetLabel() {
			return this->label_;
		}
		

	private:
		std::vector<double> image_data_;
		uint8_t label_;
	};

	class MnistDataHandler {
	public:
		MnistDataHandler();
		~MnistDataHandler();

		void LoadFeatureVector(std::string file_name);
		void LoadFeatureLabels(std::string file_name);

		void CreateData(int n_per_entry = 1);

		void SplitData();

		uint32_t ConvertToLittleEndian(const unsigned char* bytes);

		std::vector<std::shared_ptr<MnistData>>& GetRawTrainingData() {
			return this->training_data_;
		}
		std::vector<std::shared_ptr<MnistData>>& GetRawTestingData() {
			return this->testing_data_;
		}
		std::vector<std::shared_ptr<MnistData>>& GetRawValidationData() {
			return this->validation_data_;
		}

		std::tuple<std::vector<const std::vector<double>*>, std::vector<const std::vector<double>*>> GetTrainingData() {
			std::vector<const std::vector<double>*> X;
			std::vector<const std::vector<double>*> Y;

			for (auto& data : training_data_) {
				X.push_back(&data->GetImageData());
				std::vector<double>* y = new std::vector<double>(10, 0.0);
				y->at(data->GetLabel()) = 1.0;
				Y.push_back(y);
			}

			return std::make_tuple(X, Y);	
		}

		std::tuple<std::vector<const std::vector<double>*>, std::vector<int>> GetTestingData() {
			std::vector<const std::vector<double>*> X;
			std::vector<int> Y;

			for (auto& data : testing_data_) {
				X.push_back(&data->GetImageData());
				Y.push_back(data->GetLabel());
			}

			return std::make_tuple(X, Y);
		}

		std::tuple<std::vector<const std::vector<double>*>, std::vector<int>> GetValidationData() {
			std::vector<const std::vector<double>*> X;
			std::vector<int> Y;

			for (auto& data : validation_data_) {
				X.push_back(&data->GetImageData());
				Y.push_back(data->GetLabel());
			}

			return std::make_tuple(X, Y);
		}

		void PrintRandom(int n) {
			std::random_device rnd_device;
			std::mt19937 engine{ rnd_device() };
			std::uniform_int_distribution<int> dist{ 0, (int)training_data_.size()};

			for (int i = 0; i < n; i++) {
				int index = dist(engine);
				auto image_data = training_data_[index]->GetImageData();
				int label = training_data_[index]->GetLabel();

				for (int y = 0; y < 28; y++) {
					for (int x = 0; x < 28; x++) {

						if (image_data[y * 28 + x] == 1)
							std::cout << "# ";
						else if (image_data[y * 28 + x] > 0.5)
							std::cout << "o ";
						else if (image_data[y * 28 + x] >= 0.10)
							std::cout << ". ";
						else
							std::cout << "  ";
					}
					std::cout << "\n";
				}
				std::cout << "Label: " << label << "\n\n";
			}
		}
	private:
		std::vector<std::shared_ptr<MnistData>> raw_data_; // data pre-split
		std::vector<std::shared_ptr<MnistData>> training_data_;
		std::vector<std::shared_ptr<MnistData>> testing_data_;
		std::vector<std::shared_ptr<MnistData>> validation_data_;

		const double kTrainingPercentage = 0.75;
		const double kTestingPercentage = 0.2;
		const double kValidationPercentage = 0.05;
	};

}