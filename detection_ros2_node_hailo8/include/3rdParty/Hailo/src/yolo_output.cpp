/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include "yolo_output.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

std::pair<uint, float> YoloOutputLayer::get_class(const uint& row, const uint& col, const uint& anchor) const
{
	uint cls_prob, prob_max = 0;
	uint selected_class_id = 1;
	for (uint class_id = m_label_offset; class_id <= m_num_classes; class_id++)
	{
		cls_prob = get_class_prob(row, col, anchor, class_id);
		if (cls_prob > prob_max)
		{
			selected_class_id = class_id;
			prob_max          = cls_prob;
		}
	}

	return std::pair<uint, float>(selected_class_id, get_class_conf(prob_max));
}

float YoloOutputLayer::get_confidence(const uint& row, const uint& col, const uint& anchor) const
{
	uint channel     = m_pTensor->features() / NUM_ANCHORS * anchor + CONF_CHANNEL_OFFSET;
	float confidence = m_pTensor->get_full_percision(row, col, channel, m_is_uint16);
	if (m_perform_sigmoid)
		confidence = sigmoid(confidence);

	return confidence;
}

uint YoloOutputLayer::get_class_prob(const uint& row, const uint& col, const uint& anchor, const uint& class_id) const
{
	uint channel = m_pTensor->features() / NUM_ANCHORS * anchor + CLASS_CHANNEL_OFFSET + class_id - 1;
	if (m_is_uint16)
		return m_pTensor->get_uint16(row, col, channel);
	else
		return m_pTensor->get(row, col, channel);
}

float Yolov7OL::get_class_conf(const uint& prob_max) const
{
	float conf = m_pTensor->fix_scale(prob_max);
	if (m_perform_sigmoid)
		conf = sigmoid(conf);

	return conf;
}

std::pair<float, float> Yolov7OL::get_center(const uint& row, const uint& col, const uint& anchor) const
{
	float x, y = 0.0f;
	uint channel = m_pTensor->features() / NUM_ANCHORS * anchor;
	x            = (m_pTensor->get_full_percision(row, col, channel, m_is_uint16) * 2.0f - 0.5f + col) / m_width;
	y            = (m_pTensor->get_full_percision(row, col, channel + 1, m_is_uint16) * 2.0f - 0.5f + row) / m_height;

	return std::pair<float, float>(x, y);
}

std::pair<float, float> Yolov7OL::get_shape(const uint& row, const uint& col, const uint& anchor, const uint& image_width, const uint& image_height) const
{
	float w, h = 0.0f;
	uint channel = m_pTensor->features() / NUM_ANCHORS * anchor + NUM_CENTERS;
	w            = pow(2.0f * m_pTensor->get_full_percision(row, col, channel, m_is_uint16), 2.0f) * m_anchors[anchor * 2] / image_width;
	h            = pow(2.0f * m_pTensor->get_full_percision(row, col, channel + 1, m_is_uint16), 2.0f) * m_anchors[anchor * 2 + 1] / image_height;

	return std::pair<float, float>(w, h);
}
