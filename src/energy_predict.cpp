//
// Created by liyankuan on 2022/1/17.
//
#include "energy_predict.h"


//
void energy_pre::reset()//没打中大符，大符时间重置故所有参数都要重置
{
	Q << 10, 0,
	     0, 8;
	R << 0.5;
	sigma = Q;
	start_p = { -1,-1 };
	start_c = {-1,-1};
	measure_angle = 0.1;
	angle_k_1 << 0.1, 0.2;
	//start_time = -1;
	last_time = 0;
	depth = 0;
	t=0;
	dt=0;
	flip_angle = false;
	hited_count=0;
//	count = 0;
}

void energy_pre::hit_reset()//打中大符后所需的重置函数
{
    Q << 10, 0,
         0, 8;
    R << 0.5;
	sigma = Q;
	start_c = {-1,-1};
	start_p = {-1,-1};
	measure_angle = 0.1;
	angle_k_1 << 0.1, 0.2;
	last_time = 0;
	flip_angle = false;
}



cv::Point energy_pre::angle2_xy(cv::Point &now_xy, double pred_angle)
{
    cv::Point R_center = center_R.center;
    double radius = sqrt((R_center.x-Aim_armor.x)*(R_center.x-Aim_armor.x)+(R_center.y-Aim_armor.y)*(R_center.y-Aim_armor.y));
	double angle = atan2(now_xy.y - R_center.y, now_xy.x - R_center.x);
	double pre_x, pre_angle, pre_y;
	if (direct == 1)
	{
		//1为顺时针
		pre_angle = keep_pi(angle + pred_angle);
		
	}
	else
	{
		//0为逆时针
		pre_angle = keep_pi(angle - pred_angle);
	}
	pre_x = radius*cos(pre_angle) + R_center.x;
	pre_y = radius*sin(pre_angle) + R_center.y;
	
	
	
	return cv::Point(pre_x, pre_y);
}


energy_pre::energy_pre()
{
	H << 1,0;
	Q << 10, 0,
	     0, 8;
	R << 0.5;
	sigma = Q;
	F_MAT=(cv::Mat_<double>(3, 3) << 1572.95566, 0.000000000000, 631.34618, 0.000000000000, 1572.71538, 523.05524, 0.000000000000, 0.000000000000, 1.000000000000);
	C_MAT=(cv::Mat_<double>(1, 5) << -0.08780, 0.21354, -0.00000, 0.00006, 0.00000);
	
	cv::cv2eigen(F_MAT,F_EGN);
	cv::cv2eigen(C_MAT,C_EGN);
	//this->enermy_color = RED;
}




Eigen::Matrix<double,2,1> energy_pre::predict(double get_dt, bool pre_not, bool samll_energy)
{
    if (pre_not)
    {
        F << 1,dt,
             0,1;
        angle_k_1 = F*angle_k_1;
        return angle_k_1;
    }
    else
    {
        F << 1,get_dt,
             0,1;
        return F*angle_k_1;
    }
}

Eigen::Matrix<double,2,1> energy_pre::correct(Eigen::Matrix<double,1,1> &measure)
{
	sigma = F * sigma * F.transpose() + Q;
	
	K = sigma * H.transpose() * (H * sigma * H.transpose() + R).inverse();
	
	angle_k_1 = angle_k_1 + K * (measure - H * angle_k_1);
	
	sigma = (identity - K * H) * sigma;
	
	return angle_k_1;
}

double energy_pre::keep_pi(double angle)
{
	if (angle > CV_PI)
	{
		return -(CV_PI + CV_PI - angle);
	}
	else if (angle < -CV_PI)
	{
		return CV_PI + CV_PI + angle;
	}
	else
	{
		return angle;
	}
}



cv::Point energy_pre::gravity_finish(cv::Point& pps)
{
	double height;
	
	//----------用到目标点而不是预测点是因为要获取目标的距离------------
	//double depth = ap(2,0);
	printf("depth:!!!%lf\n",depth);
//	std::cout<<depth<<std::endl;
	//------------------------------------------------------------------
	
	cv::Point r_pps(pps.x/*+210*/,pps.y/*+200*/);
	Eigen::Vector3d p_pre = {(double)r_pps.x,(double)r_pps.y,1.0};
	Eigen::Vector3d ap_pre = pu_to_pc(p_pre,depth);
	
	double del_ta = pow(SPEED, 4) + 2 * 9.8 * ap_pre(1, 0) * SPEED * SPEED - 9.8 * 9.8 * depth*depth;
	double t_2 = (9.8 * ap_pre(1, 0) + SPEED * SPEED - sqrt(del_ta)) / (0.5 * 9.8 * 9.8);
	height = 0.5 * 9.8 * t_2;
	//std::cout<<"抬枪补偿:"<<height<<std::endl;
	Eigen::Vector3d ap_g = {ap_pre(0,0),ap_pre(1,0) - height,depth};
	E_pitch = atan2(ap_pre(1,0) + 0.05 - height*1., depth)/CV_PI*180.0;
	E_yaw = atan2(ap_pre(0,0) - 0.04, depth)/CV_PI*180.0;
	
	Eigen::Vector3d ap_pu = pc_to_pu(ap_g,depth);//ap_g(2,0)是距离
	
	return cv::Point((int)ap_pu(0,0)/*-210*/,(int)ap_pu(1,0)/*-200*/);
}

double energy_pre::measured(cv::Point& xy) //重写
{
    cv::Point R_center = center_R.center;
    double radius = sqrt((R_center.x-Aim_armor.x)*(R_center.x-Aim_armor.x)+(R_center.y-Aim_armor.y)*(R_center.y-Aim_armor.y));
	int offset_x = R_center.x - start_c.x;
	int offset_y = R_center.y - start_c.y;
	cv::Point _start_p = {start_p.x+offset_x,start_p.y+offset_y};
	double dis_st = sqrt((xy.x - _start_p.x) * (xy.x - _start_p.x) + (xy.y - _start_p.y) * (xy.y - _start_p.y)) / 2;
	//防止出现无穷大数据，因为asin函数如果入参大于1就会出现无穷大数据
	if (dis_st > radius)
	{
		dis_st = radius;
	}
	
	double angle = 2 * asin(dis_st/radius);
	
	if (angle > 2.98)
	{
		flip_angle = true;
	}
	if (flip_angle)
	{
		angle = 2 * CV_PI - angle;
	}
	measure_angle = angle;
	//std::cout <<"观测值"<< measure_angle << std::endl;
	if (measure_angle < 0)
	{
		measure_angle = 0.10;
	}
	
	return measure_angle;
}

//bool energy_pre::cal_dela_angle()
//{
//    double dela_dis = sqrt((Aim_armor.x-last_dt_p.x)*(Aim_armor.x-last_dt_p.x)+(Aim_armor.y-last_dt_p.y)*(Aim_armor.y-last_dt_p.y));
//    double angle = (2*asin((dela_dis/2.0)/radius))/CV_PI*180.0;
//    if (angle >= 69)
//    {
//        return true;
//    }
//    else
//    {
//        return false;
//    }
//}

void energy_pre::get_direct(cv::Point &now)
{
    cv::Point R_center = center_R.center;
    double symbol = (now.y - R_center.y) * (last_p.y - R_center.y);
    double now_ang = atan2(now.y - R_center.y, now.x - R_center.x);
    double five_ang = atan2(last_p.y - R_center.y, last_p.x - R_center.x);
    if (symbol < 0)//是否在x轴的交界处
        {
        if ((now_ang - five_ang) < 0)
        {
            direct = abs(now_ang) > CV_PI / 2 && abs(five_ang) > CV_PI / 2 ? 1 : 0;
        }
        else
        {
            direct = abs(now_ang) > CV_PI / 2 && abs(five_ang) > CV_PI / 2 ? 0 : 1;
        }
        }
    else
    {
        if ((now_ang - five_ang) < 0)
        {
            direct = 0;
        }
        else
        {
            direct = 1;
        }
    }


}
