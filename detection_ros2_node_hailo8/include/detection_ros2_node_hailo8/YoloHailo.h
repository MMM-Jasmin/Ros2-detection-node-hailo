#pragma once

#include "HailoPower.h"
#include "Utils.h"
#include "hailo_objects.hpp"
#include "hailo_tensors.hpp"
#include "hailomat.hpp"
#include "yolo_postprocess.hpp"

#include <iostream>

static const uint32_t YOLOV7_IMAGE_WIDTH  = 640;
static const uint32_t YOLOV7_IMAGE_HEIGHT = 640;

DEFINE_EXCEPTION(YoloHailoException)

using HailoDetectionPtrs = std::vector<HailoDetectionPtr>;

class YoloHailo
{
	DISABLE_COPY_ASSIGN_MOVE(YoloHailo)

	class FeatureData
	{
	public:
		FeatureData(uint32_t buffers_size, hailo_vstream_info_t vstream_info) :
			m_buffer(buffers_size), m_vstream_info(vstream_info)
		{
		}

		std::vector<uint8_t> m_buffer;
		hailo_vstream_info_t m_vstream_info;
	};

public:
	YoloHailo(const std::string &hefFile, const std::string &classFile, const uint32_t width, const uint32_t height, const float &threshold = 0.3f, const bool &autoLoad = true) :
		m_hefFile(hefFile), m_width(width), m_height(height), m_threshold(threshold)
	{
		// parseClassFile(classFile);
		if (autoLoad)
			Initialze();
	}

	~YoloHailo()
	{
		StopPowerMeasuring();
	}

	hailo_status Initialze()
	{
		std::cout << "Setting up inference ... " << std::flush;

		auto vdevice = hailort::VDevice::create();
		if (!vdevice)
		{
			std::cerr << "Failed create vdevice, status = " << vdevice.status() << std::endl;
			return vdevice.status();
		}

		m_pVdevice = std::move(vdevice.value());

		auto hef = hailort::Hef::create(m_hefFile);
		if (!hef)
			return hef.status();

		auto configureParams = hef->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
		if (!configureParams)
			return configureParams.status();

		auto networkGroups = m_pVdevice->configure(hef.value(), configureParams.value());
		if (!networkGroups)
			return networkGroups.status();

		if (1 != networkGroups->size())
		{
			std::cerr << "Invalid amount of network groups" << std::endl;
			return HAILO_INTERNAL_FAILURE;
		}

		m_networkGroup = std::move(networkGroups->front());

		auto vstreams = hailort::VStreamsBuilder::create_vstreams(*m_networkGroup, true, HAILO_FORMAT_TYPE_AUTO);
		if (!vstreams)
		{
			std::cerr << "Failed creating vstreams " << vstreams.status() << std::endl;
			return vstreams.status();
		}

		m_vstreams = std::move(vstreams.value());

		m_pPower = std::make_shared<HailoPower>(m_pVdevice);

		// Create features data to be used for post-processing
		std::size_t num_features = m_vstreams.second.size();

		m_features.reserve(num_features);
		for (size_t i = 0; i < num_features; i++)
			m_features.emplace_back(std::make_shared<FeatureData>(static_cast<uint32_t>(m_vstreams.second[i].get_frame_size()), m_vstreams.second[i].get_info()));

		m_pYoloInitParams = std::make_shared<Yolov7Params>(m_threshold);

		std::cout << "Done" << std::endl;

		m_initialized = true;

		return HAILO_SUCCESS;
	}

	std::size_t GetClassCount()const
	{
		return m_pYoloInitParams->getLabels().size();
	}

	void StartPowerMeasuring()
	{
		if (m_pPower)
			m_pPower->startPowerMeasurement();
	}

	void PrintPowerMeasuring() const
	{
		if (m_pPower)
			m_pPower->getPowerMeasurement();
	}

	float GetAveragePower(const std::size_t idx = 0) const
	{
		return m_pPower->getAveragePower(idx);
	}

	void StopPowerMeasuring()
	{
		if (m_pPower)
			m_pPower->stopPowerMeasurement();
	}

	HailoDetectionPtrs Infer(const HailoRGBMat &image)
	{
		if (!m_initialized)
		{
			std::cerr << "YoloHailo is not initialized" << std::endl;
			return HailoDetectionPtrs();
		}

		return runInference(image);
	}

	HailoDetectionPtrs Infer(const cv::Mat &image)
	{
		if (!m_initialized)
		{
			std::cerr << "YoloHailo is not initialized" << std::endl;
			return HailoDetectionPtrs();
		}

		return runInference(image);
	}

private:
	void parseClassFile(const std::string &classFile)
	{
		m_classes.clear();
		std::ifstream f(classFile);
		if (!f.is_open())
			throw(YoloHailoException(string_format("Failed to load class file: %s", classFile.c_str())));

		std::string line;
		while ((std::getline(f, line)))
			m_classes.push_back(line);
	}

	HailoDetectionPtrs runInference(const HailoRGBMat &image)
	{
		return runInference(image.get_mat(), image.get_name());
	}

	HailoDetectionPtrs runInference(const cv::Mat &image, const std::string &fileName = "")
	{
		cv::Mat scaledMat;
		cv::Mat rgbMat;
		if ((YOLOV7_IMAGE_WIDTH != image.cols) || (YOLOV7_IMAGE_HEIGHT != image.rows))
			cv::resize(image, scaledMat, cv::Size(YOLOV7_IMAGE_WIDTH, YOLOV7_IMAGE_WIDTH));
		else
			scaledMat = image;

		cv::cvtColor(scaledMat, rgbMat, cv::COLOR_BGR2RGB);

		// WRITE
		hailo_status status = m_vstreams.first.front().write(hailort::MemoryView(scaledMat.data, scaledMat.total() * scaledMat.elemSize()));
		if (HAILO_SUCCESS != status)
		{
			std::cerr << "Failed writing to device data of image '" << fileName << "'. Got status = " << status << std::endl;
			return HailoDetectionPtrs();
		}

		// READ
		for (size_t i = 0; i < m_vstreams.second.size(); i++)
		{
			auto &buffer = m_features[i]->m_buffer;
			status       = m_vstreams.second[i].read(hailort::MemoryView(buffer.data(), buffer.size()));

			if (HAILO_SUCCESS != status)
			{
				std::cerr << "Failed reading with status = " << status << std::endl;
				return HailoDetectionPtrs();
			}
		}

		// POST-PROCESS
		// Gather the features into HailoTensors in a HailoROIPtr
		HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
		for (const auto &feature : m_features)
			roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t *>(feature->m_buffer.data()), feature->m_vstream_info));

		// Perform the actual postprocess
		yolov7(roi, m_pYoloInitParams);

		return hailo_common::get_hailo_detections(roi);
	}

private:
	std::string m_hefFile;
	float m_threshold;
	std::vector<std::string> m_classes = std::vector<std::string>();
	uint32_t m_width;
	uint32_t m_height;

	bool m_initialized = false;

	std::unique_ptr<hailort::VDevice> m_pVdevice                    = nullptr;
	std::shared_ptr<hailort::ConfiguredNetworkGroup> m_networkGroup = nullptr;

	std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> m_vstreams = std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>>();

	std::vector<std::shared_ptr<FeatureData>> m_features = std::vector<std::shared_ptr<FeatureData>>();
	YoloParamsPtr m_pYoloInitParams                      = nullptr;

	std::shared_ptr<HailoPower> m_pPower = nullptr;
};
