#include "detection_node_hailo8.hpp"

#include <hailo/hailort.hpp>
#include "Timer.h"
#include "YoloHailo.h"
#include "hailomat.hpp"

#include "SORT.h"

#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

const double ONE_SECOND            = 1000.0; // One second in milliseconds

/**
 * @brief Contructor.
 */
DetectionNodeHailo8::DetectionNodeHailo8(const std::string &name) : Node(name, rclcpp::NodeOptions().use_intra_process_comms(false)) 
{
	this->declare_parameter("rotation", 0);
	this->declare_parameter("debug", false);
	this->declare_parameter("topic", "");
	this->declare_parameter("image_size", 640);
	this->declare_parameter("print_detections", true);
	this->declare_parameter("print_fps", true);
	this->declare_parameter("det_topic", "test/det");
	this->declare_parameter("fps_topic", "test/fps");
	this->declare_parameter("power_topic", "test/watt");
	this->declare_parameter("max_fps", 30.0f);
	this->declare_parameter("qos_sensor_data", true);
	this->declare_parameter("qos_history_depth", 10);
	 
    this->declare_parameter("DLA_CORE", 0);
    this->declare_parameter("USE_FP16", true);
    this->declare_parameter("ONNX_FILE", "");
    this->declare_parameter("CONFIG_FILE", "");
    this->declare_parameter("ENGINE_FILE" , "");
    this->declare_parameter("CLASS_FILE", "");
    this->declare_parameter("DETECT_STR", "");
    this->declare_parameter("AMOUNT_STR", "");
    this->declare_parameter("FPS_STR", "");
    this->declare_parameter("YOLO_VERSION", 4);
    this->declare_parameter("YOLO_TINY", true);
    this->declare_parameter("YOLO_THRESHOLD", 0.3);	
	this->declare_parameter("YOLOV7_HEF_FILE","/opt/dev/DL_Models/yolo_object/model/yolov7.hef");
	

	callback_handle_ = this->add_on_set_parameters_callback(std::bind(&DetectionNodeHailo8::parametersCallback, this, std::placeholders::_1));
}

rcl_interfaces::msg::SetParametersResult DetectionNodeHailo8::parametersCallback(const std::vector<rclcpp::Parameter> &parameters)
{

	for (const auto &param: parameters){
		if (param.get_name() == "max_fps")
			m_maxFPS = param.as_double();
	}

	rcl_interfaces::msg::SetParametersResult result;
	result.successful = true;
    result.reason = "success";
	return result;
}

/**
 * @brief Initialize image node.
 */
void DetectionNodeHailo8::init() {


	int YOLO_VERSION, DLA_CORE, qos_history_depth, image_size;
	bool USE_FP16, YOLO_TINY, qos_sensor_data;
	float YOLO_THRESHOLD;
	std::string ONNX_FILE, CONFIG_FILE, ENGINE_FILE, CLASS_FILE, YOLOV7_HEF_FILE, ros_topic, det_topic, fps_topic, power_topic;

	std::cout << "-- get ros config variables --" << std::endl;

	// needed only for init
	// get ros configuration
	this->get_parameter("topic", ros_topic);
	this->get_parameter("det_topic", det_topic);
	this->get_parameter("fps_topic", fps_topic);
	this->get_parameter("power_topic", power_topic);
	this->get_parameter("image_size", image_size);
	
	// get Yolo configuration
 	this->get_parameter("DLA_CORE", DLA_CORE);
    this->get_parameter("USE_FP16", USE_FP16);
    this->get_parameter("ONNX_FILE", ONNX_FILE);
    this->get_parameter("CONFIG_FILE", CONFIG_FILE);
    this->get_parameter("ENGINE_FILE" , ENGINE_FILE);
    this->get_parameter("CLASS_FILE", CLASS_FILE);
    this->get_parameter("YOLO_VERSION", YOLO_VERSION);
    this->get_parameter("YOLO_TINY", YOLO_TINY);
    this->get_parameter("YOLO_THRESHOLD", YOLO_THRESHOLD);
	this->get_parameter("YOLOV7_HEF_FILE",YOLOV7_HEF_FILE);

	// some things needs to be member
	this->get_parameter("max_fps", m_maxFPS);
	this->get_parameter("DETECT_STR", m_DETECT_STR);
    this->get_parameter("AMOUNT_STR", m_AMOUNT_STR);
    this->get_parameter("FPS_STR", m_FPS_STR);
	this->get_parameter("rotation", m_image_rotation);
	this->get_parameter("print_detections", m_print_detections);
	this->get_parameter("print_fps", m_print_fps);
	this->get_parameter("qos_sensor_data", qos_sensor_data);
	this->get_parameter("qos_history_depth", qos_history_depth);

	std::cout << "-- init hailo8 --" << std::endl;


	m_pYoloHailo8 = new YoloHailo(YOLOV7_HEF_FILE, CLASS_FILE, YOLO_THRESHOLD);
	m_pYoloHailo8->StartPowerMeasuring();

	m_pSortTrackers = new SORT[m_pYoloHailo8->GetClassCount()];
	////// Initialize SORT tracker for each class
	for (std::size_t i = 0; i < m_pYoloHailo8->GetClassCount(); i++)
		m_pSortTrackers[i] = SORT(30, 5);

	m_lastTrackings.clear();

	m_elapsedTime = 0;
	m_timer.Start();

	std::cout << "-- subscribe to : " << ros_topic <<  " --" << std::endl;

	if(qos_sensor_data){
		std::cout << "using ROS2 qos_sensor_data" << std::endl;
		m_qos_profile = rclcpp::SensorDataQoS();
	}

	m_qos_profile = m_qos_profile.keep_last(qos_history_depth);
	//m_qos_profile = m_qos_profile.lifespan(std::chrono::milliseconds(500));
	m_qos_profile = m_qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
	//m_qos_profile = m_qos_profile.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
	//m_qos_profile = m_qos_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
	
	m_qos_profile_sysdef = m_qos_profile_sysdef.keep_last(qos_history_depth);
	//m_qos_profile_sysdef = m_qos_profile_sysdef.lifespan(std::chrono::milliseconds(500));
	m_qos_profile_sysdef = m_qos_profile_sysdef.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
	//m_qos_profile_sysdef = m_qos_profile_sysdef.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
	//m_qos_profile_sysdef = m_qos_profile_sysdef.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
	
	m_image_small_subscription = this->create_subscription<sensor_msgs::msg::Image>( ros_topic, m_qos_profile, std::bind(&DetectionNodeHailo8::imageSmallCallback, this, std::placeholders::_1));
	//cv::namedWindow(m_window_name_image_small, cv::WINDOW_AUTOSIZE);

	std::cout << "-- create topics for publishing --" << std::endl;

	m_detection_publisher   = this->create_publisher<std_msgs::msg::String>(det_topic, m_qos_profile_sysdef);
	m_fps_publisher    		= this->create_publisher<std_msgs::msg::String>(fps_topic, m_qos_profile_sysdef);
	m_power_publisher    	= this->create_publisher<std_msgs::msg::String>(power_topic, m_qos_profile_sysdef);

	std::cout << "+==========[ init done ]==========+" << std::endl;
}


/**
 * @brief Callback function for reveived image message.
 * @param img_msg Received image message
 */
void DetectionNodeHailo8::imageSmallCallback(sensor_msgs::msg::Image::SharedPtr img_msg) {

	cv::Size image_size(static_cast<int>(img_msg->width), static_cast<int>(img_msg->height));
	cv::Mat color_image(image_size, CV_8UC3, (void *)img_msg->data.data(), cv::Mat::AUTO_STEP);

	ProcessNextFrame(color_image);
	ProcessDetections();
	
	m_frameCnt++;
	CheckFPS(&m_frameCnt);
}

void DetectionNodeHailo8::ProcessDetections( )
{
	bool changed                        = false;
	//const ::YoloResults& results = m_yoloResults;

	std::map<uint32_t, TrackingObjects> trackingDets;

	for (const YoloHailo::YoloResult& res : m_yoloHailoResults)
	{
		uint32_t id  = res.classID - 1;
		float x      = res.x;
		float y      = res.y;
		float width  = res.w;
		float height = res.h;

		if (x < 0.0f) x = 0.0f;
		if (y < 0.0f) y = 0.0f;
		if (width > 1.0f) width = 1.0f;
		if (height > 1.0f) height = 1.0f;

		trackingDets.try_emplace(id, TrackingObjects());
		trackingDets[id].push_back({ { x ,y , width, height }, static_cast<uint32_t>(std::round(res.classProb * 100)), res.label});
	}

	TrackingObjects trackers;
	TrackingObjects dets;

	for (std::size_t i = 0; i < m_pYoloHailo8->GetClassCount(); i++)
	{
		if (trackingDets.count(i))
			dets = trackingDets[i];
		else
			dets = TrackingObjects();
			TrackingObjects t = m_pSortTrackers[i].Update(dets);
		trackers.insert(std::end(trackers), std::begin(t), std::end(t));
	}

	if (trackers.size() != m_lastTrackings.size())
		changed = true;
	else
	{
		for (const auto& [idx, obj] : enumerate(trackers))
		{
			if (m_lastTrackings[idx] != obj)
			{
				changed = true;
				break;
			}
		}
	}

	if (changed || (m_framesSincePublish > 30)){
		printDetections(trackers);
		m_framesSincePublish = 0;

	}
	m_framesSincePublish++;
}

void DetectionNodeHailo8::ProcessNextFrame(cv::Mat &img)
{
	if (!img.empty()){
		m_yoloHailoResults = m_pYoloHailo8->Infer(img);
	}
} 

BBox DetectionNodeHailo8::toCenter(const BBox& bBox)
{
	// x_y = center
	float h = bBox.height;
	float w = bBox.width;
	float x = bBox.x + (w / 2);
	float y = bBox.y + (h / 2);
	return BBox(x, y, w, h);
}

void DetectionNodeHailo8::printDetections(const TrackingObjects& trackers)
{
	std::stringstream str("");
	str << string_format("{\"%s\": [", m_DETECT_STR.c_str());

	for (const auto& [i, t] : enumerate(trackers))
	{
		BBox centerBox = toCenter(t.bBox);
		str << string_format("{\"TrackID\": %i, \"name\": \"%s\", \"center\": [%.3f,%.3f], \"w_h\": [%.3f,%.3f]}", t.trackingID, t.name.c_str(), roundf(centerBox.x*1000.0f)/1000.0f , roundf(centerBox.y*1000.0f)/1000.0f, roundf(centerBox.width*1000.0f)/1000.0f, roundf(centerBox.height*1000.0f)/1000.0f);
		// Prevent a trailing ',' for the last element
		if (i + 1 < trackers.size()) str << ", ";
	}

	m_lastTrackings = trackers;

	str << string_format("], \"%s\": %llu }", m_AMOUNT_STR.c_str(), m_lastTrackings.size());

	auto message = std_msgs::msg::String();
	message.data = str.str();
	try{
		m_detection_publisher->publish(message);
	}
	catch (...) {
		RCLCPP_INFO(this->get_logger(), "hmm publishing dets has failed!! ");
	}

	if (m_print_detections)
		RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
	
}

void DetectionNodeHailo8::CheckFPS(uint64_t* pFrameCnt)
	{
		m_timer.Stop();

		double minFrameTime = 1000.0 / m_maxFPS;
		double itrTime      = m_timer.GetElapsedTimeInMilliSec();
		double fps;

		m_elapsedTime += itrTime;

		fps = 1000 / (m_elapsedTime / (*pFrameCnt));

		if (m_elapsedTime >= ONE_SECOND)
		{
			PrintFPS(fps, itrTime);

			*pFrameCnt    = 0;
			m_elapsedTime = 0;
		}

		m_timer.Start();
	}

void DetectionNodeHailo8::PrintFPS(const float fps, const float itrTime)
{
		
	std::stringstream str("");

	if (fps == 0.0f)
			str << string_format("{\"%s\": 0.0}", m_FPS_STR.c_str());
	else
		str << string_format("{\"%s\": %.2f, \"lastCurrMSec\": %.2f, \"maxFPS\": %.2f, \"%s\": %llu }", m_FPS_STR.c_str(), fps, itrTime, m_maxFPS, m_AMOUNT_STR.c_str(), m_lastTrackings.size());

	auto message = std_msgs::msg::String();
	message.data = str.str();

	auto power_message = std_msgs::msg::String();
	power_message.data = std::to_string(m_pYoloHailo8->GetAveragePower());
	
	try{
		m_fps_publisher->publish(message);
		m_power_publisher->publish(power_message);
	}
  	catch (...) {
    	RCLCPP_INFO(this->get_logger(), "m_fps_publisher: hmm publishing dets has failed!! ");
  	}

		
	if (m_print_fps)
		RCLCPP_INFO(this->get_logger(), message.data.c_str());

}
