/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <algorithm>
#include <sstream>
#include <typeinfo>
#include <vector>

#include "nms.hpp"
#include "yolo_postprocess.hpp"

#if __GNUC__ > 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

class YoloPost
{
public:
	virtual ~YoloPost() = default;
	YoloPost(std::map<uint8_t, std::string> dataset, float detection_threshold, float iou_threshold, uint max_boxes) :
		m_max_boxes(max_boxes),
		m_detection_thr(detection_threshold),
		m_iou_thr(iou_threshold),
		m_dataset(dataset){};

	std::vector<HailoDetection> decode()
	{
		std::vector<HailoDetection> objects;
		objects.reserve(m_max_boxes);
		for (const auto &layer : m_layers)
			extract_boxes(layer, objects);

		common::nms(objects, m_iou_thr);
		if (objects.size() > m_max_boxes)
		{
			HailoBBox bbox(0, 0, 1, 1);
			HailoDetection empty_detection(bbox, "None", 0.0);
			objects.resize(m_max_boxes, empty_detection);
		}

		return objects;
	}

	const uint &get_num_classes() const
	{
		return m_layers[0]->getNumClasses();
	}

	/**
     * @brief Extract the boxes of generic yolo output layer.
     *
     * @param[in] image_size Network's input image width/height.
     * @param[in] thr Postprocess threshold.
     * @param[out] objects Reference to vector of detections.
     */
	void extract_boxes(const std::shared_ptr<YoloOutputLayer> &layer, std::vector<HailoDetection> &objects);

protected:
	std::vector<std::shared_ptr<YoloOutputLayer>> m_layers;
	uint m_max_boxes;
	float m_detection_thr;
	float m_iou_thr;
	uint m_image_width;
	uint m_image_height;
	std::map<uint8_t, std::string> m_dataset;
};

void YoloPost::extract_boxes(const std::shared_ptr<YoloOutputLayer> &layer, std::vector<HailoDetection> &objects)
{
	uint class_id = 0;
	float x, y, h, w, confidence, class_confidence = 0.0f;
	float xmin, ymin = 0.0f;

	for (uint row = 0; row < layer->getHeight(); ++row)
	{
		for (uint col = 0; col < layer->getWidth(); ++col)
		{
			for (uint anchor = 0; anchor < layer->NUM_ANCHORS; ++anchor)
			{
				confidence = layer->get_confidence(row, col, anchor);
				if (confidence < m_detection_thr)
					continue;

				std::tie(class_id, class_confidence) = layer->get_class(row, col, anchor);
				// Final confidence: box confidence * class probability
				confidence = confidence * class_confidence;

				if (confidence > m_detection_thr)
				{
					std::tie(x, y) = layer->get_center(row, col, anchor);
					std::tie(w, h) = layer->get_shape(row, col, anchor, m_image_width, m_image_height);
					// Get the top left corner of the object.
					xmin = (x - (w / 2.0f));
					ymin = (y - (h / 2.0f));
					objects.push_back(HailoDetection(HailoBBox(xmin, ymin, w, h), class_id, m_dataset[class_id], confidence));
				}
			}
		}
	}
}

class Yolov7 : public YoloPost
{
public:
	Yolov7(HailoROIPtr roi, const YoloParamsPtr pParams) :
		YoloPost(pParams->getLabels(), pParams->getDetectionThreshold(), pParams->getIouThreshold(), pParams->getMaxBoxes()),
		m_tensors(roi->get_tensors())
	{
		if (m_tensors.size() > 0)
		{
			bool sigmoid = (pParams->getOutputActivation() == "sigmoid");
			sort(m_tensors.begin(), m_tensors.end(), [](const HailoTensorPtr &a, const HailoTensorPtr &b)
				 { return a->size() < b->size(); });

			m_image_width  = m_tensors[0]->width() * 32;
			m_image_height = m_tensors[0]->height() * 32;
			m_layers.reserve(m_tensors.size());

			for (std::size_t i = 0; i < m_tensors.size(); i++)
			{
				hailo_format_type_t format = m_tensors[i]->vstream_info().format.type;
				m_layers.push_back(std::make_shared<Yolov7OL>(m_tensors[i], pParams->getAnchorsVec()[i], sigmoid, pParams->getLabelOffset(), format == HAILO_FORMAT_TYPE_UINT16));
			}
		}

		pParams->check_params_logic(get_num_classes());
	};

	virtual ~Yolov7() = default;

private:
	std::vector<HailoTensorPtr> m_tensors;
};

void yolov7(HailoROIPtr roi, const YoloParamsPtr pParams)
{
	auto post          = Yolov7(roi, pParams);
	auto detections    = post.decode();
	hailo_common::add_detections(roi, detections);
}

void YoloParams::check_params_logic(const uint &num_classes_tensors) const
{
	if (m_labels.size() - 1 != num_classes_tensors)
	{
		std::ostringstream oss;
		oss << "config class labels do not match output tensors! config labels size: " << m_labels.size() - 1 << " tensors num classes: " << num_classes_tensors << std::endl;
		throw std::runtime_error(oss.str());
	}
}
