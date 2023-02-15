/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file hailo_objects.hpp
 * @authors Hailo
 **/

#pragma once

#include "hailo_tensors.hpp"
#include <map>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>

#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define CLIP(x) (CLAMP(x, 0, 255))
#define NULL_CLASS_ID (-1)

typedef enum
{
    HAILO_ROI,
    HAILO_CLASSIFICATION,
    HAILO_DETECTION,
    HAILO_LANDMARKS,
    HAILO_TILE,
    HAILO_UNIQUE_ID,
    HAILO_MATRIX,
    HAILO_DEPTH_MASK,
    HAILO_CLASS_MASK,
    HAILO_CONF_CLASS_MASK,
    HAILO_USER_META
} hailo_object_t;

static float assure_normal(float num)
{
    if ((num > 1.0f) || (num < 0.0))
    {
        throw std::invalid_argument("Number should be between 0.0 to 1.0.");
    }
    return num;
}

/**
 * @brief HailoBBox - Represents a bounding box.
 * Takes 4 float arguments representing the bounding box (normalized).
 * The first 2 arguments are the x and y of the minimum point (Top Left corner).
 * The other 2 arguments are the width and height of the box respectivly.
 * All arguments are normalized to the picture they relate to.
 */
struct HailoBBox
{
protected:
    float m_xmin;
    float m_ymin;
    float m_width;
    float m_height;

public:
    /**
     * @brief Construct a new Hailo BBox object
     *
     * @param xmin normalized xmin position
     * @param ymin normalized ymin position
     * @param width normalized width of bounding box
     * @param height normalized height of bounding box
     */
    HailoBBox(float xmin, float ymin, float width, float height) : m_xmin(xmin), m_ymin(ymin), m_width(width), m_height(height){};

    const float xmin() const { return m_xmin; }
    const float ymin() const { return m_ymin; }
    const float width() const { return m_width; }
    const float height() const { return m_height; }
    const float xmax() const { return m_xmin + m_width; }
    const float ymax() const { return m_ymin + m_height; }
};

/**
 * @brief Represents an object that is a usable output after postprocessing.
 * An abstract class for all objects to inherit from.
 */
class HailoObject
{
protected:
    std::shared_ptr<std::mutex> mutex;

public:
    // Constructor
    HailoObject()
    {
        mutex = std::make_shared<std::mutex>();
    };
    // Destructor
    virtual ~HailoObject() = default;
    HailoObject &operator=(const HailoObject &other) = default;
    HailoObject &operator=(HailoObject &&other) noexcept = default;
    HailoObject(HailoObject &&other) noexcept = default;
    HailoObject(const HailoObject &other) = default;

    /**
     * @brief Get the type object
     *
     * @return hailo_object_t - The type of the object.
     */
    virtual hailo_object_t get_type() = 0;
};

using HailoObjectPtr = std::shared_ptr<HailoObject>;

/**
 * @brief Represents a HailoObject that can hold other objects.
 *  for example a face detection can hold landmarks or age classification, gender classification etc...
 */
class HailoMainObject : public HailoObject, public std::enable_shared_from_this<HailoMainObject>
{
protected:
    std::vector<HailoObjectPtr> m_sub_objects;
    std::map<std::string, HailoTensorPtr> m_tensors;

public:
    HailoMainObject()
    {
        mutex = std::make_shared<std::mutex>();
    };
    virtual ~HailoMainObject() = default;
    HailoMainObject(HailoMainObject &&other) noexcept : HailoObject(other), m_sub_objects(std::move(other.m_sub_objects)){};
    HailoMainObject(const HailoMainObject &other) : HailoObject(other), m_sub_objects(other.m_sub_objects){};
    HailoMainObject &operator=(const HailoMainObject &other) = default;
    HailoMainObject &operator=(HailoMainObject &&other) noexcept = default;

    /**
     * @brief Add an object to the main object.
     *
     * @param obj Object to add.
     */
    void add_object(const HailoObjectPtr obj)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_sub_objects.emplace_back(obj);
    };

    /**
     * @brief Add a tensor to the main object.
     *
     * @param tensor Tensor to add.
     */
    void add_tensor(HailoTensorPtr tensor)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_tensors.emplace(tensor->name(), tensor);
    };

    /**
     * @brief Remove a HailoObject from the MainObject
     *
     * @param obj  -  HailoObjectPtr
     *        The object to remove
     */
    void remove_object(HailoObjectPtr obj)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_sub_objects.erase(std::remove(m_sub_objects.begin(), m_sub_objects.end(), obj), m_sub_objects.end());
    };

    /**
     * @brief Remove a HailoObject from the MainObject
     *
     * @param index  -  uint
     *        The index of the object to remove
     */
    void remove_object(uint index)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_sub_objects.erase(m_sub_objects.begin() + index);
    };

    /**
     * @brief Get a tensor from this main object.
     *
     * @param name Tensor's name to get,
     * @return HailoTensorPtr - A tensor.
     */
    HailoTensorPtr get_tensor(std::string name)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        auto itr = m_tensors.find(name);
        if (itr == m_tensors.end())
        {
            throw std::invalid_argument("No tensor with name " + name);
        }
        return itr->second;
    };

    /**
     * @brief Checks whether there are tensors attached to this main object
     *
     * @return true when there are tensors in this main object.
     * @return false when there are no tensors in this main object.
     */
    bool has_tensors()
    {
        return !m_tensors.empty();
    };

    /**
     * @brief Get a vector of the tensors attached to this main object.
     *
     * @return std::vector<HailoTensorPtr>
     */
    std::vector<HailoTensorPtr> get_tensors()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        std::vector<HailoTensorPtr> _tensors;
        _tensors.reserve(m_tensors.size());
        for (auto &tensor_pair : m_tensors)
        {
            _tensors.emplace_back(tensor_pair.second);
        }
        return _tensors;
    };

    std::map<std::string, HailoTensorPtr> get_tensors_by_name()
    {
        std::map<std::string, HailoTensorPtr> tensors_by_name;
        auto tensors = get_tensors();
        for (auto tensor : tensors)
        {
            tensors_by_name[tensor->name()] = tensor;
        }
        return tensors_by_name;
    }

    /**
     * @brief Clear all tensors attached to this main object.
     *
     */
    void clear_tensors()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_tensors.clear();
    }

    /**
     * @brief Get the objects attached to this main object
     *
     * @return std::vector<HailoObjectPtr>
     */
    std::vector<HailoObjectPtr> get_objects()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_sub_objects;
    }

    /**
     * @brief Get the objects of a given type, attached to this main object.
     *
     * @param type The type of object to get.
     * @return std::vector<HailoObjectPtr>
     */
    std::vector<HailoObjectPtr> get_objects_typed(hailo_object_t type)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        std::vector<HailoObjectPtr> filtered_subobjects;
        for (auto &obj : m_sub_objects)
        {
            if (obj->get_type() == type)
            {
                filtered_subobjects.emplace_back(obj);
            }
        }
        return filtered_subobjects;
    }

    /**
     * @brief Removes all the objects of a given type, attached to this main object.
     *
     * @param type The type of object to get.
     */
    void remove_objects_typed(hailo_object_t type)
    {
        for (auto obj : this->get_objects_typed(type))
        {
            this->remove_object(obj);
        }
    }
};
using HailoMainObjectPtr = std::shared_ptr<HailoMainObject>;

/**
 * @brief Represents an ROI (Region Of Interest).
 * Part of an image, can hold other objects. Mostly inherited by other objects but isn't abstract.
 * Can represents the whole image by giving the right HailoBBox.
 */
class HailoROI : public HailoMainObject
{
protected:
    HailoBBox m_bbox;         // A bounding box - the normalized position of this region of interest.
    HailoBBox m_scaling_bbox; // A bounding box to scale by - x offset, y offset, width factor, height factor
public:
    HailoROI(HailoBBox bbox) : m_bbox(bbox), m_scaling_bbox(HailoBBox(0.0, 0.0, 1.0, 1.0)){};
    virtual ~HailoROI() = default;
    HailoROI(HailoROI &&other) noexcept : HailoMainObject(other), m_bbox(std::move(other.m_bbox)), m_scaling_bbox(std::move(other.m_scaling_bbox)){};
    HailoROI(const HailoROI &other) : HailoMainObject(other), m_bbox(other.m_bbox), m_scaling_bbox(std::move(other.m_scaling_bbox)){};
    HailoROI &operator=(const HailoROI &other) = default;
    HailoROI &operator=(HailoROI &&other) noexcept = default;

    std::shared_ptr<HailoROI> shared_from_this()
    {
        return std::dynamic_pointer_cast<HailoROI>(HailoMainObject::shared_from_this());
    }

    virtual hailo_object_t get_type()
    {
        return HAILO_ROI;
    }

    /**
     * @brief Add an object to the main object.
     *
     * @param obj Object to add.
     */
    void add_object(const HailoObjectPtr obj)
    {
        std::shared_ptr<HailoROI> possible_roi = std::dynamic_pointer_cast<HailoROI>(obj);
        if (nullptr != possible_roi)
            possible_roi->set_scaling_bbox(this->get_bbox());
        HailoMainObject::add_object(obj);
    };

    /**
     * @brief Get the bbox of this ROI
     *
     * @return HailoBBox&
     */
    HailoBBox &get_bbox()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_bbox;
    }

    /**
     * @brief Set the bbox of this ROI
     *
     * @param bbox
     */
    void set_bbox(HailoBBox bbox)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_bbox = std::move(bbox);
    }

    /**
     * @brief Get the scaling bbox of this ROI
     *
     * @return HailoBBox&
     */
    HailoBBox &get_scaling_bbox()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_scaling_bbox;
    }

    /**
     * @brief Set the scaling bbox of this ROI
     *
     * @param bbox
     */
    void set_scaling_bbox(HailoBBox bbox)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        float new_xmin = (m_scaling_bbox.xmin() * bbox.width()) + bbox.xmin();
        float new_ymin = (m_scaling_bbox.ymin() * bbox.height()) + bbox.ymin();
        float new_width = m_scaling_bbox.width() * bbox.width();
        float new_height = m_scaling_bbox.height() * bbox.height();
        HailoBBox new_scale = HailoBBox(new_xmin, new_ymin, new_width, new_height);
        m_scaling_bbox = std::move(new_scale);
    }

    /**
     * @brief Clear the scaling bbox of this ROI
     *
     */
    void clear_scaling_bbox()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_scaling_bbox = HailoBBox(0.0, 0.0, 1.0, 1.0);
    }
};
using HailoROIPtr = std::shared_ptr<HailoROI>;

/**
 * @brief Represents a detection in a ROI. Inherits from HailoROI.
 *
 */
class HailoDetection : public HailoROI
{
protected:
    float m_confidence;  // Confidence of the detection.
    std::string m_label; // The label of detection, e.g. "Horse", "Monkey", "Tiger" for type "Animals".
    int m_class_id;      // Class id, initialized to -1 if missing.
public:
    /**
     * @brief Construct a new New Hailo Detection object
     *
     * @param bbox HailoBBox - a bounding box representing the region of interest in the frame.
     * @param label std::string what the detection is.
     * @param confidence The confidence of the detection.
     * @note class id is set to -1.
     */
    HailoDetection(HailoBBox bbox, const std::string &label, float confidence) : HailoROI(bbox), m_confidence(assure_normal(confidence)), m_label(label), m_class_id(NULL_CLASS_ID){};
    /**
     * @brief Construct a new New Hailo Detection object
     *
     * @param bbox HailoBBox - a bounding box representing the region of interest in the frame.
     * @param class_id The detection's class id, if theres any.
     * @param label std::string what the detection is.
     * @param confidence The confidence of the detection.
     */
    HailoDetection(HailoBBox bbox, int class_id, const std::string &label, float confidence) : HailoROI(bbox), m_confidence(assure_normal(confidence)), m_label(label), m_class_id(class_id){};

    // Move constructor
    HailoDetection(HailoDetection &&other) noexcept : HailoROI(other),
                                                      m_confidence(assure_normal(other.m_confidence)),
                                                      m_label(std::move(other.m_label)),
                                                      m_class_id(other.m_class_id){};
    // Copy constructor
    HailoDetection(const HailoDetection &other) : HailoROI(other),
                                                  m_confidence(assure_normal(other.m_confidence)),
                                                  m_label(std::move(other.m_label)),
                                                  m_class_id(other.m_class_id){};
    virtual ~HailoDetection() = default;

    // Move assignment
    HailoDetection &operator=(HailoDetection &&other) noexcept
    {
        if (this != &other)
        {
            HailoROI::operator=(other);
            m_confidence = assure_normal(other.m_confidence);
            m_class_id = other.m_class_id;
            m_label = std::move(other.m_label);
        }
        return *this;
    };
    // Copy assignment
    HailoDetection &operator=(const HailoDetection &other)
    {
        if (this != &other)
        {
            HailoROI::operator=(other);
            m_confidence = assure_normal(other.m_confidence);
            m_class_id = other.m_class_id;
            m_label = other.m_label;
        }
        return *this;
    };
    // Overload comparison operators
    bool operator<(const HailoDetection &other) const
    {
        return this->m_confidence < other.m_confidence;
    }

    bool operator>(const HailoDetection &other) const
    {
        return this->m_confidence > other.m_confidence;
    }

    virtual hailo_object_t get_type()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return HAILO_DETECTION;
    }

    std::shared_ptr<HailoObject> clone()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return std::make_shared<HailoDetection>(*this);
    }

    // Getters of DetectionObject.

    float get_confidence()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_confidence;
    }
    void set_confidence(float conf)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_confidence = conf;
    }
    std::string get_label()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_label;
    }
    int get_class_id()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_class_id;
    }
};
using HailoDetectionPtr = std::shared_ptr<HailoDetection>;
