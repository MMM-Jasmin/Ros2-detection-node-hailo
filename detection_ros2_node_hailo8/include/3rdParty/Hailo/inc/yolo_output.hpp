/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#pragma once
#include "hailo_objects.hpp"
#include <cmath>
#include <iostream>

/**
 * @brief Base class to represent OutputLayer of Yolo networks.
 *
 */
class YoloOutputLayer
{
public:
	static const uint NUM_ANCHORS          = 3;
	static const uint NUM_CENTERS          = 2;
	static const uint NUM_SCALES           = 2;
	static const uint NUM_CONF             = 1;
	static const uint CONF_CHANNEL_OFFSET  = NUM_CENTERS + NUM_SCALES;
	static const uint CLASS_CHANNEL_OFFSET = CONF_CHANNEL_OFFSET + NUM_CONF;

	YoloOutputLayer(const uint& width, const uint& height, const uint& num_of_classes, const std::vector<int>& anchors, const bool& perform_sigmoid,
					const int& label_offset, const bool& is_uint16, const HailoTensorPtr pTensor = nullptr) :
		m_width(width),
		m_height(height),
		m_num_classes(num_of_classes),
		m_anchors(anchors),
		m_label_offset(label_offset),
		m_perform_sigmoid(perform_sigmoid),
		m_is_uint16(is_uint16),
		m_pTensor(pTensor){};

	virtual ~YoloOutputLayer() = default;

	/**
     * @brief Get the class object
     *
     * @param row
     * @param col
     * @param anchor
     * @return std::pair<uint, float> class id and class probability.
     */
	std::pair<uint, float> get_class(const uint& row, const uint& col, const uint& anchor) const;
	/**
     * @brief Get the confidence object
     *
     * @param row
     * @param col
     * @param anchor
     * @return float
     */
	virtual float get_confidence(const uint& row, const uint& col, const uint& anchor) const;
	/**
     * @brief Get the center object
     *
     * @param row
     * @param col
     * @param anchor
     * @return std::pair<float, float> pair of x,y of the center of this prediction.
     */
	virtual std::pair<float, float> get_center(const uint& row, const uint& col, const uint& anchor) const = 0;
	/**
     * @brief Get the shape object
     *
     * @param row
     * @param col
     * @param anchor
     * @param image_width
     * @param image_height
     * @return std::pair<float, float> pair of w,h of the shape of this prediction.
     */
	virtual std::pair<float, float> get_shape(const uint& row, const uint& col, const uint& anchor, const uint& image_width, const uint& image_height) const = 0;

	const uint& getWidth() const
	{
		return m_width;
	}

	const uint& getHeight() const
	{
		return m_height;
	}

	const uint& getNumClasses() const
	{
		return m_num_classes;
	}

protected:
	/**
     * @brief Get the class channel object
     *
     * @param anchor
     * @param channel
     * @return uint
     */
	virtual uint get_class_prob(const uint& row, const uint& col, const uint& anchor, const uint& class_id) const;
	/**
     * @brief Get the class conf object
     *
     * @param prob_max
     * @return float
     */
	virtual float get_class_conf(const uint& prob_max) const = 0;

	static float sigmoid(const float& x)
	{
		// returns the value of the sigmoid function f(x) = 1/(1 + e^-x)
		return 1.0f / (1.0f + expf(-x));
	}

	static uint num_classes(const uint& channels)
	{
		return (channels / NUM_ANCHORS) - CLASS_CHANNEL_OFFSET;
	}

	uint m_width;
	uint m_height;
	uint m_num_classes;
	std::vector<int> m_anchors;
	int m_label_offset;

	bool m_perform_sigmoid;
	bool m_is_uint16;
	HailoTensorPtr m_pTensor;
};

class Yolov7OL : public YoloOutputLayer
{
public:
	Yolov7OL(HailoTensorPtr tensor, std::vector<int> anchors, bool perform_sigmoid, int label_offset, bool is_uint16) :
		YoloOutputLayer(tensor->width(), tensor->height(), num_classes(tensor->features()), anchors, false, label_offset, is_uint16, tensor){};

	virtual float get_class_conf(const uint& prob_max) const;
	virtual std::pair<float, float> get_center(const uint& row, const uint& col, const uint& anchor) const;
	virtual std::pair<float, float> get_shape(const uint& row, const uint& col, const uint& anchor, const uint& image_width, const uint& image_height) const;
};
