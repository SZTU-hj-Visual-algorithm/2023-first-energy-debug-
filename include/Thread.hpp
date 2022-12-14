#pragma once
#include "camera.h"
#include "ArmorDetector.hpp"
#include "KAl.h"
#include <opencv2/core/cvstd.hpp>
#include "CRC_Check.h"
#include"serialport.h"
#include <thread>
#include <mutex>
#include<string>
#include <iostream>
#include "energy_predict.h"
#include <opencv2/opencv.hpp>
#include "robot_state.h"

extern pthread_mutex_t mutex_new; 
extern pthread_cond_t cond_new; 
extern pthread_mutex_t mutex_kal;
extern pthread_cond_t cond_kal;

extern bool is_kal;
extern bool is_start;         
extern cv::Mat src;      

extern cv::Mat  quan_src;
extern float quan_ab_pitch;
extern float quan_ab_yaw;
extern float quan_ab_roll;
extern float quan_speed;

void* Build_Src(void* PARAM);
void* Armor_Kal(void* PARAM);
void* Kal_predict(void* PARAM);
