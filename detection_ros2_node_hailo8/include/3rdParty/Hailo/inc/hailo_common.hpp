/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/* __BEGIN_DECLS should be used at the beginning of your declarations,
   so that C++ compilers don't mangle their names.  Use __END_DECLS at
   the end of C declarations. */

#pragma once
#include "hailo_objects.hpp"

namespace hailo_common
{
inline void add_object(HailoROIPtr roi, const HailoObjectPtr obj)
{
	roi->add_object(obj);
}

inline HailoDetectionPtr add_detection(HailoROIPtr roi, const HailoBBox& bbox, const std::string& label, const float& confidence, const int& class_id = NULL_CLASS_ID)
{
	HailoDetectionPtr detection = std::make_shared<HailoDetection>(bbox, class_id, label, confidence);
	detection->set_scaling_bbox(roi->get_bbox());
	add_object(roi, detection);
	return detection;
}

inline void add_detections(HailoROIPtr roi, const std::vector<HailoDetection>& detections)
{
	for (const auto& det : detections)
		add_object(roi, std::make_shared<HailoDetection>(det));
}

inline std::vector<HailoDetectionPtr> get_hailo_detections(const HailoROIPtr roi)
{
	std::vector<HailoObjectPtr> objects = roi->get_objects_typed(HAILO_DETECTION);
	std::vector<HailoDetectionPtr> detections;

	for (const auto& obj : objects)
		detections.emplace_back(std::dynamic_pointer_cast<HailoDetection>(obj));

	return detections;
}

/**
     * Flatten HailoBBox with parent HailoBBox.
     * re scales each bbox values (x,y,width,height min/max values) to match the parent scale.
     *
     * @param[in] bbox    HailoBBox, target bbox to flatten.
     * @param[in] parent_bbox  HailoBBox, parent bbox - to scale to.
     * @return void.
     */
inline HailoBBox create_flattened_bbox(const HailoBBox& bbox, const HailoBBox& parent_bbox)
{
	float xmin = parent_bbox.xmin() + bbox.xmin() * parent_bbox.width();
	float ymin = parent_bbox.ymin() + bbox.ymin() * parent_bbox.height();

	float width  = bbox.width() * parent_bbox.width();
	float height = bbox.height() * parent_bbox.height();

	return HailoBBox(xmin, ymin, width, height);
}
} // namespace hailo_common
