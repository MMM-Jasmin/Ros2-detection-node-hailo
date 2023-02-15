#pragma once

#include <hailo/hailort.hpp>

#include <iostream>
#include <vector>

#include "Utils.h"

DEFINE_EXCEPTION(HailoPowerException)

class HailoPower
{
	const hailo_sampling_period_e SAMPLING_PERIOD                   = HAILO_SAMPLING_PERIOD_1100US;
	const hailo_averaging_factor_e AVERAGE_FACTOR                   = HAILO_AVERAGE_FACTOR_256;
	const hailo_dvm_options_t DVM_OPTION                            = HAILO_DVM_OPTIONS_AUTO; // For current measurement over EVB - pass DVM explicitly (see hailo_dvm_options_t)
	const hailo_measurement_buffer_index_t MEASUREMENT_BUFFER_INDEX = HAILO_MEASUREMENT_BUFFER_INDEX_0;

public:
	HailoPower(const std::unique_ptr<hailort::VDevice> &pVdevice)
	{
		auto physDevs = pVdevice->get_physical_devices();
		if (!physDevs)
			throwError("Failed to get physical devices", physDevs.status());

		m_physDevs = physDevs.value();
		m_measurementResults = std::vector<hailo_power_measurement_data_t>(m_physDevs.size());
	}

	 ~HailoPower()
	 {
		if(m_running) stopPowerMeasurement();
	 }

	void printMeasurementResults(hailort::Device &device, const hailo_power_measurement_data_t &result) const
	{
		auto id = device.get_dev_id();

		std::cout << "Device" << std::string(id) << ":" << std::endl;
		std::cout << "  Power measurement" << std::endl;
		std::cout << "    Minimum value: " << result.min_value << "W" << std::endl;
		std::cout << "    Average value: " << result.average_value << "W" << std::endl;
		std::cout << "    Maximum value: " << result.max_value << "W" << std::endl;
	}

	void startPowerMeasurement()
	{
		for (auto &physDev : m_physDevs)
		{
			hailo_status status = physDev.get().stop_power_measurement();
			if (HAILO_SUCCESS != status)
				throwError("Failed stopping former measurement", status);

			status = physDev.get().set_power_measurement(MEASUREMENT_BUFFER_INDEX, DVM_OPTION, HAILO_POWER_MEASUREMENT_TYPES__POWER);
			if (HAILO_SUCCESS != status)
				throwError("Failed setting measurement params", status);

			status = physDev.get().start_power_measurement(AVERAGE_FACTOR, SAMPLING_PERIOD);
			if (HAILO_SUCCESS != status)
				throwError("Failed starting measurement", status);
		}

		m_running = true;
	}

	void getPowerMeasurement() const
	{
		for (std::size_t i = 0; i < m_physDevs.size(); i++)
		{
			const auto &physDev = m_physDevs.at(i);
			if (m_running)
			{
				auto measurementResult = physDev.get().get_power_measurement(MEASUREMENT_BUFFER_INDEX, false);
				if (!measurementResult)
					throwError("Failed to get measurement results", measurementResult.status());

				printMeasurementResults(physDev.get(), measurementResult.value());
			}
			else
				printMeasurementResults(physDev.get(), m_measurementResults.at(i));
		}
	}

	float getAveragePower(const std::size_t idx = 0) const
	{
		if(m_physDevs.size() <= idx) return 0.0f;
		if (m_running)
		{
			auto measurementResult = m_physDevs.at(idx).get().get_power_measurement(MEASUREMENT_BUFFER_INDEX, false);
			if (!measurementResult)
				throwError("Failed to get measurement results", measurementResult.status());

			return measurementResult.value().average_value;
		}

		return m_measurementResults.at(idx).average_value;
	}

	void stopPowerMeasurement()
	{
		m_running = false;

		for (std::size_t i = 0; i < m_physDevs.size(); i++)
		{
			const auto &physDev = m_physDevs.at(i);
			hailo_status status = physDev.get().stop_power_measurement();
			if (HAILO_SUCCESS != status)
				throwError("Failed stopping measurement", status);

			auto measurementResult = physDev.get().get_power_measurement(MEASUREMENT_BUFFER_INDEX, true);
			if (!measurementResult)
				throwError("Failed to get measurement results", measurementResult.status());

			m_measurementResults.at(i) = measurementResult.value();

			printMeasurementResults(physDev.get(), m_measurementResults.at(i));
		}
	}

private:
	static inline void throwError(const std::string &msg, const hailo_status &status)
	{
		throw(HailoPowerException(string_format("%s - Status=%d", msg.c_str(), status)));
	}

private:
	std::vector<std::reference_wrapper<hailort::Device>> m_physDevs  = std::vector<std::reference_wrapper<hailort::Device>>();
	std::vector<hailo_power_measurement_data_t> m_measurementResults = std::vector<hailo_power_measurement_data_t>();

	bool m_running = false;
};
