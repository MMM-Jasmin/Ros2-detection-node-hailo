/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file overlay/common.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-01-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <opencv2/opencv.hpp>
#include "hailo_common.hpp"
#include "hailo_objects.hpp"

class HailoMat
{
protected:
    uint m_height;
    uint m_width;
    uint m_stride;
    int m_line_thickness;
    int m_font_thickness;
    cv::Rect get_bounding_rect(HailoBBox bbox, uint channel_width, uint channel_height)
    {
        cv::Rect rect;
        uint width = channel_width;
        uint height = channel_height;
        rect.x = CLAMP(bbox.xmin() * width, 0, width);
        rect.y = CLAMP(bbox.ymin() * height, 0, height);
        rect.width = CLAMP(bbox.width() * width, 0, width - rect.x);
        rect.height = CLAMP(bbox.height() * height, 0, height - rect.y);
        return rect;
    }

public:
    HailoMat(uint height, uint width, uint stride, int line_thickness = 1, int font_thickness = 1) : m_height(height),
                                                                                                     m_width(width),
                                                                                                     m_stride(stride),
                                                                                                     m_line_thickness(line_thickness),
                                                                                                     m_font_thickness(font_thickness){};
    HailoMat() : m_height(0), m_width(0), m_stride(0), m_line_thickness(0), m_font_thickness(0){};
    virtual ~HailoMat() = default;
    const uint& width() const { return m_width; }
    const uint& height() const { return m_height; }
    virtual cv::Mat &get_mat() = 0;
    virtual const cv::Mat &get_mat() const = 0;
    virtual void draw_rectangle(cv::Rect rect, const cv::Scalar color) = 0;
    virtual void draw_text(std::string text, cv::Point position, double font_scale, const cv::Scalar color) = 0;
    virtual void draw_line(cv::Point point1, cv::Point point2, const cv::Scalar color, int thickness, int line_type) = 0;
    virtual void draw_ellipse(cv::Point center, cv::Size axes, double angle, double start_angle, double end_angle, const cv::Scalar color, int thickness) = 0;
    /*
     * @brief Crop ROIs from the mat, note the present implementation is valid
     *        for interlaced formats. Planar formats such as NV12 should override.
     * 
     * @param crop_roi 
     *        The roi to crop from this mat.
     * @return cv::Mat 
     *         The cropped mat.
     */
    virtual cv::Mat crop(HailoROIPtr crop_roi)
    {
        auto bbox = hailo_common::create_flattened_bbox(crop_roi->get_bbox(), crop_roi->get_scaling_bbox());
        cv::Rect rect = get_bounding_rect(bbox, m_width, m_height);
        cv::Mat cropped_cv_mat = get_mat()(rect);
        return cropped_cv_mat;
    }
};

class HailoRGBMat : public HailoMat
{
protected:
    cv::Mat m_mat;
    std::string m_name;

public:
    HailoRGBMat(uint8_t *buffer, uint height, uint width, uint stride, int line_thickness = 1, int font_thickness = 1, std::string name = "HailoRGBMat") : HailoMat(height, width, stride, line_thickness, font_thickness)
    {
        m_name = name;
        m_mat = cv::Mat(m_height, m_width, CV_8UC3, buffer, m_stride);
    };
    HailoRGBMat(cv::Mat mat, std::string name, int line_thickness = 1, int font_thickness = 1)
    {
        m_mat = mat;
        m_name = name;
        m_height = mat.rows;
        m_width = mat.cols;
        m_stride = mat.step;
        m_line_thickness = line_thickness;
        m_font_thickness = font_thickness;
    }
    virtual cv::Mat &get_mat()
    {
        return m_mat;
    }
    virtual const cv::Mat &get_mat() const
    {
        return m_mat;
    }
    virtual const std::string& get_name() const
    {
        return m_name;
    }
	
    virtual void draw_rectangle(cv::Rect rect, const cv::Scalar color)
    {
        cv::rectangle(m_mat, rect, color, m_line_thickness);
    }
    virtual void draw_text(std::string text, cv::Point position, double font_scale, const cv::Scalar color)
    {
        cv::putText(m_mat, text, position, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, m_font_thickness);
    }
    virtual void draw_line(cv::Point point1, cv::Point point2, const cv::Scalar color, int thickness, int line_type)
    {
        cv::line(m_mat, point1, point2, color, thickness, line_type);
    }
    virtual void draw_ellipse(cv::Point center, cv::Size axes, double angle, double start_angle, double end_angle, const cv::Scalar color, int thickness)
    {
        cv::ellipse(m_mat, center, axes, angle, start_angle, end_angle, color, thickness);
    }
    virtual ~HailoRGBMat()
    {
        m_mat.release();
    }
};
