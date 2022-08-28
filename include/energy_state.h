//
// Created by 蓬蒿浪人 on 2022/7/17.
//

#ifndef CLION_EXPERI_ENERGY_STATE_H
#define CLION_EXPERI_ENERGY_STATE_H

#include <opencv2/opencv.hpp>
#include "robot_state.h"
#include <Eigen/Dense>
//using namespace cv;
using namespace std;

struct energy_inf
{
    cv::Point re_aim;
    cv::RotatedRect c_rect;

};

class energy:public robot_state
{
public:
    cv::Point2f dst_p[4] = {cv::Point2f(0,0),cv::Point2f(0,30),cv::Point2f(60,30),cv::Point2f(60,0)};
    cv::Mat src;
    cv::Mat center_mat;
    cv::Mat F_MAT,C_MAT;
    vector<cv::Mat> warp_vec;
    cv::Vec2d real_xy;
    int hit_count = 0;
    int hited_count = 0;
    int change_aim;
    double depth = 0;
    deque<double> distances;
    Eigen::Matrix3d F_EGN;
    Eigen::Matrix<double,1,5> C_EGN;
    void show_all_dst();
    energy_inf detect_aim();
    void adjust_bin(cv::Mat &_src);
    Eigen::Vector3d pnp_get(cv::Rect &c_rect);
    void get_ap(Eigen::Vector3d &real_c,cv::Point &Aim,cv::RotatedRect &c_rect);
    double depth_filter(deque<double> &dis);
    void make_c_safe(cv::Rect &line);

};


#endif //CLION_EXPERI_ENERGY_STATE_H
