/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#pragma once
#include "coco_eighty.hpp"
#include "hailo_common.hpp"
#include "hailo_objects.hpp"
#include "yolo_output.hpp"

const float IOU_THRESHOLD           = 0.45f;
const std::string OUTPUT_ACTIVATION = "none";
const int32_t LABEL_OFFSET          = 1;
const int32_t MAX_BOXES             = 200;

const std::vector<std::vector<int>> YOLO_V7_ANCHORS_VEC = { { 142, 110, 192, 243, 459, 401 }, { 36, 75, 76, 55, 72, 146 }, { 12, 16, 19, 36, 40, 28 } };

class YoloParams
{
public:
	YoloParams(const float& detection_threshold, const std::map<std::uint8_t, std::string>& labels, const float& iou_threshold,
			   const uint& max_boxes, const std::vector<std::vector<int>>& anchors_vec, const std::string& output_activation,
			   const int& label_offset) :
		m_iou_threshold(iou_threshold),
		m_detection_threshold(detection_threshold),
		m_labels(labels),
		m_max_boxes(max_boxes),
		m_anchors_vec(anchors_vec),
		m_output_activation(output_activation),
		m_label_offset(label_offset)
	{
	}

	void check_params_logic(const uint& num_classes_tensors) const;

	float getIouThreshold() const
	{
		return m_iou_threshold;
	}

	float getDetectionThreshold() const
	{
		return m_detection_threshold;
	}

	const std::map<std::uint8_t, std::string>& getLabels() const
	{
		return m_labels;
	}

	uint getMaxBoxes() const
	{
		return m_max_boxes;
	}

	const std::vector<std::vector<int>>& getAnchorsVec() const
	{
		return m_anchors_vec;
	}

	const std::string& getOutputActivation() const
	{
		return m_output_activation;
	}

	const int& getLabelOffset() const
	{
		return m_label_offset;
	}

protected:
	float m_iou_threshold;
	float m_detection_threshold;
	std::map<std::uint8_t, std::string> m_labels;
	uint m_max_boxes = MAX_BOXES;
	std::vector<std::vector<int>> m_anchors_vec;
	std::string m_output_activation; // can be "none" or "sigmoid"
	int m_label_offset;
};

using YoloParamsPtr = std::shared_ptr<YoloParams>;

class Yolov7Params : public YoloParams
{
public:
	Yolov7Params(const float& detection_threshold, const std::map<std::uint8_t, std::string>& labels = common::coco_eighty,
				 const float& iou_threshold = IOU_THRESHOLD, const uint& max_boxes = MAX_BOXES, const std::vector<std::vector<int>>& anchors_vec = YOLO_V7_ANCHORS_VEC,
				 const std::string& output_activation = OUTPUT_ACTIVATION, const int& label_offset = LABEL_OFFSET) :
		YoloParams(detection_threshold, labels, iou_threshold, max_boxes, anchors_vec, output_activation, label_offset)
	{
	}
};

using Yolov7ParamsPtr = std::shared_ptr<Yolov7Params>;

void yolov7(HailoROIPtr roi, const YoloParamsPtr pParams);
