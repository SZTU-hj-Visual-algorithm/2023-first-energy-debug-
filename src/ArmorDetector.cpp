/*******************************************************************************************************************
Copyright 2017 Dajiang Innovations Technology Co., Ltd (DJI)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files(the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and / or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
*******************************************************************************************************************/

#include "ArmorDetector.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <queue>
#include "string""
#include <cmath>

#define SHOW_ALL_CANDIDATE
//#define SHOW_CANDIDATE_IMG
#define SHOW_Binary_IMG
//#define SHOW_MATCHED_RECT_ARMOR
//#define COUT_LOG
//#define CLASSIFICATION
#define SafeRect(rect, max_size) {if (makeRectSafe(rect, max_size) == false) continue;}

using namespace cv;
using namespace std;

void ArmorDetector::setImage(const cv::Mat & src){
	_size = src.size();  // 一直是1280*720的
	// 注意这个_res_last是一个旋转矩形
	const cv::Point & last_result = _res_last.center;
    float last_lr_rate = _res_last_matched.lr_rate;
//    cout << last_lr_rate << endl;
	// 如果上一次的目标没了，源图就是输入的图
	// 并且搜索的ROI矩形（_dect_rect）就是整个图像
	if (last_result.x == 0 || last_result.y == 0) {
		_src = src;
		_dect_rect = Rect(0, 0, src.cols, src.rows);
	}
	else{
		// 如果上一次的目标没有丢失的话，用直立矩形包围上一次的旋转矩形
		Rect rect = _res_last.boundingRect();

		// _para.max_light_delta_h 是左右灯柱在水平位置上的最大差值，像素单位
		int max_half_w = _para.max_light_delta_h * 1.3;
		int max_half_h = 300;

        //TODO: 截取ROI图像需要优化，以适应反陀螺识别（已优化）

		// 截图的区域大小。太大的话会把45度识别进去
		double scale_w = 2;
		double scale_h = 2;
        int lu_x_offset = 0;
        int rd_x_offset = 0;
        // 根据灯条高度比设置偏移量
        if(last_lr_rate>1)
            lu_x_offset = 6 *( pow(last_lr_rate - 1, 0.6) + 0.09) * rect.width;
        else
            rd_x_offset = 6 * (pow(1 - last_lr_rate, 0.6) + 0.15) * rect.width;
//        cout << "lu: " << lu_x_offset << " rd: " << rd_x_offset << endl;
        int w = int(rect.width * scale_w);
		int h = int(rect.height * scale_h);

        Point center = last_result;

		int x = std::max(center.x - w - lu_x_offset, 0);
		int y = std::max(static_cast<int>(center.y + h), 0);
		Point lu = Point(x, y);  /* point left up */

		x = std::min(center.x + w + rd_x_offset, src.cols);
		y = std::min(static_cast<int>(center.y + h), src.rows);
		Point rd = Point(x, y);  /* point right down */


		// 构造出矩形找到了搜索的ROI区域
		_dect_rect = Rect(lu, rd);

		// 为false说明矩形是空的，所以继续搜索全局像素
		//FIXME：感觉这里会有点bug
		if (!makeRectSafe(_dect_rect, src.size())){
			_res_last = cv::RotatedRect();
			_dect_rect = Rect(0, 0, src.cols, src.rows);
			_src = src;
		}
		else
			// 如果ROI矩形合法的话就用此ROI
			src(_dect_rect).copyTo(_src);
	}

	//==========================上面已经设置好了真正处理的原图============================

	// 下面是在敌方通道上二值化进行阈值分割
	// _max_color是红蓝通道相减之后的二值图像
	///////////////////////////// begin /////////////////////////////////////////////
	/**
	 * 预处理其实是最重要的一步，这里有HSV和RGB两种预处理的思路，其实大致都差不多
	 * 只不过在特定场合可以选择特定的预处理方式
	 * 例如HSV的话可以完全过滤掉日光灯的干扰，但是耗时较大
	 */


	int thres_max_color_red = 46;
	int thres_max_color_blue = 146;

	_max_color = cv::Mat(_src.size(), CV_8UC1, cv::Scalar(0));


	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
	// 敌方颜色为红色时
    EnemyColor enermy_color = RED;
	if (enermy_color == RED){

        // TODO：验证该代码段
        ////这里初步猜测是用来排除日光灯的影响的
        ////但经过实测，如果降低相机的对焦清晰度，或是用其他的方法把原图的清晰度降低也可以排除日光灯的干扰
        ////但降低对焦清晰度不知道会不会影响远距离的识别
		//Mat thres_whole;
		//cvtColor(_src,thres_whole,CV_BGR2GRAY);

		/*
		if (sentry_mode)
			threshold(thres_whole,thres_whole,33,255,THRESH_BINARY);
		else
			threshold(thres_whole,thres_whole,60,255,THRESH_BINARY);
        //imshow("thresh_whole", thres_whole);  //测试亮度
		 */

        // TODO:优化颜色分离，二值图获取
        // 手动二值化
		int src_data = _src.rows * _src.cols;
		auto* img_data = (uchar*)_src.data;
		auto* img_data_binary = (uchar*)_max_color.data;

		for (size_t i = 0; i < src_data; i++)
		{
			if (*(img_data + 2) - *img_data > thres_max_color_red)
				*img_data_binary = 255;
            img_data += 3;
            img_data_binary++;
		}

		/*
        Mat color;
        subtract(splited[2], splited[0], color);
        threshold(color, color, thres_max_color_red, 255, THRESH_BINARY); // red
		//imshow("color", color);   //查看颜色筛选过后的效果
        */

		//_max_color = _max_color & thres_whole;  // _max_color获得了清晰的二值图

		dilate(_max_color, _max_color, element);
	}
    // 敌方颜色是蓝色时
	else {
        //TODO：验证该代码段
/*		Mat thres_whole;
		vector<Mat> splited;
		split(_src, splited);
		cvtColor(_src,thres_whole,CV_BGR2GRAY);
		if (sentry_mode)
		  threshold(thres_whole,thres_whole,60,255,THRESH_BINARY);
		else
		  threshold(thres_whole,thres_whole,128,255,THRESH_BINARY);
		//imshow("thres_whole", thres_whole);
		*/

        // TODO:优化颜色分离，二值图获取
		int srcdata = _src.rows * _src.cols;
		auto* Imgdata = (uchar*)_src.data;
		auto* Imgdata_binary = (uchar*)_max_color.data;

		for (size_t i = 0; i < srcdata; i++)
		{
			if (*Imgdata - *(Imgdata + 2) > thres_max_color_blue)
				*Imgdata_binary = 255;
			Imgdata += 3;
			Imgdata_binary++;
		}

		/*
        subtract(splited[0], splited[2], _max_color);
		threshold(_max_color, _max_color, thres_max_color_blue, 255, THRESH_BINARY);// blue
        //imshow("color", _max_color);
        */

		//_max_color = _max_color & thres_whole;  // _max_color获得了清晰的二值图

		dilate(_max_color, _max_color, element);
	}

	////////////////////////////// end /////////////////////////////////////////
#ifdef SHOW_Binary_IMG
	cv::imshow("_max_color", _max_color);
#endif

}

vector<matched_rect> ArmorDetector::findTarget() {
	vector<matched_rect> match_rects;  // 储存合适灯条的向量
	vector<vector<Point2i>> contours_max;  // 储存全部轮廓
    //// 在红蓝/蓝红通道相减的二值图中寻找轮廓
	findContours(_max_color, contours_max, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<RotatedRect> RectFirstResult;  // 灯条筛选结果
	for (auto & i : contours_max){
        //// 用最小面积矩形拟合轮廓，找出符合筛选条件范围的轮廓
        RotatedRect rrect = minAreaRect(i);
        // 获取灯条矩形长短边
		double max_rrect_len = MAX(rrect.size.width, rrect.size.height);
		double min_rrect_len = MIN(rrect.size.width, rrect.size.height);

		/////////////////////////////// 单根灯条的条件 //////////////////////////////////////
		// 角度要根据实际进行略微改动
		bool if1 = (fabs(rrect.angle) < 30.0 && rrect.size.height > rrect.size.width); // 往左,指现在偏向左
		bool if2 = (fabs(rrect.angle) > 60.0 && rrect.size.width > rrect.size.height); // 往右，指现在偏向右
		bool if3 = max_rrect_len > _para.min_light_height; // 灯条的最小长度
		bool if4;
		if (!base_mode) // 吊射基地时条件不同
			if4 = (max_rrect_len / min_rrect_len >= 1.1) && (max_rrect_len / min_rrect_len < 15); // 灯条的长宽比
		else
			if4 = (max_rrect_len / min_rrect_len >= 9.9) && (max_rrect_len / min_rrect_len < 30); // 灯条的长宽比
		// 筛除横着的以及太小的旋转矩形 (本来是45的，后来加成60)
		if ((if1 || if2) && if3 && if4)
		{
			RectFirstResult.push_back(rrect);
		}
	}

	//// 少于两根灯条就认为没有匹配到
	if (RectFirstResult.size() >= 2) {

		// 将旋转矩形从左到右排序
		sort(RectFirstResult.begin(), RectFirstResult.end(),
			 [](RotatedRect& a1, RotatedRect& a2) {
				 return a1.center.x < a2.center.x; });

		///////////////////////////////////// 匹配灯条的条件 //////////////////////////////////////////////////////
        Point2f _pt[4], pt[4];
        //传入两个点，算出两点线段与x正半轴的角度
        auto ptangle = [](const Point2f& p1, const Point2f& p2) {
            return fabs(atan2(p2.y - p1.y, p2.x - p1.x) * 180.0 / CV_PI);  // 浮点数反正切函数atan2() 返回弧度制角度，转为角度值
        };

        //// 两两比较，有符合条件的就组成一个目标旋转矩形
		for (size_t i = 0; i < RectFirstResult.size() - 1; ++i) {
			const RotatedRect& rect_i = RectFirstResult[i];
			const Point2f & center_i = rect_i.center;  // 灯条中心坐标
			float xi = center_i.x;
			float yi = center_i.y;

			float len_i = MAX(rect_i.size.width, rect_i.size.height);  // 灯条长边
			float angle_i = fabs(rect_i.angle);
            // TODO:灯条矩形顶点顺序待确认
            /*_pt opencv 4.0
                * 1 2
                * 0 3
                * */
            rect_i.points(_pt);


            // TODO：验证pt
            /*pt
                * 0 2
                * 1 3
                * */
            // 往右斜的灯条
            if (angle_i > 45.0) {
				pt[0] = _pt[3];
				pt[1] = _pt[0];
			}
            // 往左斜的灯条
			else {
				pt[0] = _pt[2];
				pt[1] = _pt[3];
			}
            // TODO:优化灯条匹配
			for (size_t j = i + 1; j < RectFirstResult.size(); j++) {
				const RotatedRect& rect_j = RectFirstResult[j];
				const Point2f & center_j = rect_j.center;
				float xj = center_j.x;
				float yj = center_j.y;
				float len_j = MAX(rect_j.size.width, rect_j.size.height);
				float angle_j = fabs(rect_j.angle);
				float delta_h = xj - xi;  // 两灯条距离
				float lr_rate = len_i / len_j; // 高度比
				float angle_abs;

				rect_j.points(_pt);
				if (angle_j > 45.0)
				{
					pt[2] = _pt[2];
					pt[3] = _pt[1];
				}
				else {
					pt[2] = _pt[1];
					pt[3] = _pt[0];
				}

				if (angle_i > 45.0 && angle_j < 45.0) { // 八字 / \   //
                    angle_abs = 90.0 - angle_i + angle_j;    //可能是越小越好，两个小方框的差值
				}
				else if (angle_i <= 45.0 && angle_j >= 45.0) { // 倒八字 \ /
                    angle_abs = 90.0 - angle_j + angle_i;
				}
				else {
					if (angle_i > angle_j) angle_abs = angle_i - angle_j; // 在同一边
					else angle_abs = angle_j - angle_i;
				}

				//// 如果装甲板符合匹配条件，则添加到候选装甲板向量中
				bool condition1 = delta_h > _para.min_light_delta_h && delta_h < _para.max_light_delta_h;
				bool condition2 = MAX(len_i, len_j) >= 113 ? abs(yi - yj) < 166\
                    && abs(yi - yj) < 1.66 * MAX(len_i, len_j) :
								  abs(yi - yj) < _para.max_light_delta_v\
                    && abs(yi - yj) < 1.2 * MAX(len_i, len_j); // && abs(yi - yj) < MIN(leni, lenj)
				bool condition3 = lr_rate < _para.max_lr_rate;
                bool condition5 = lr_rate > _para.min_lr_rate;

                bool condition4;
				if (!base_mode)
                    // 区分哨兵模式和步兵模式
					condition4 = sentry_mode ? angle_abs < 25 : angle_abs < 15 - 5;
				else
					condition4 = angle_abs > 25 && angle_abs < 55;

//				Point test_center = Point((xi + xj) / 2, (yi + yj) / 2);  // 装甲板中心

                //// 四种情况均成立，将两个灯条组成候选旋转矩形
                if (condition1 && condition2 && condition3 && condition4 && condition5)
				{
                    RotatedRect obj_rect = boundingRRect(rect_i, rect_j);
                    ////对符合条件的旋转矩形进行初步筛选
                    //筛选掉中间包含一个灯条的矩形
                    if (Contain(obj_rect,RectFirstResult,i,j))
				    {
                        continue;
                    }
					double w = obj_rect.size.width;
					double h = obj_rect.size.height;
					wh_ratio = w / h;

					// 装甲板长宽比不符，基地模式不受长宽比的限制
					if (!base_mode) {
						if (wh_ratio > _para.max_wh_ratio || wh_ratio < _para.min_wh_ratio)
							continue;
					}

					// 将初步匹配到的结构体信息push进入vector向量
					match_rects.push_back(matched_rect{ obj_rect, lr_rate, angle_abs });

#ifdef SHOW_MATCHED_RECT_ARMOR
                    Mat rect_show;
                    _src.copyTo(rect_show);
					Point2f vertice[4];
					obj_rect.points(vertice);
					for (int k = 0; k < 4; k++)
						line(_src, vertice[k], vertice[(k + 1) % 4], Scalar(255, 255, 255), 2);
#endif
				}
			}
		}
	}
	return match_rects;
}



cv::RotatedRect ArmorDetector::chooseTarget(const std::vector<matched_rect> & match_rects, const cv::Mat & src) {
	// 如果没有两个灯条围成一个目标矩形就返回一个空旋转矩形
	if (match_rects.empty()){
		_is_lost = true;  // (未使用)
		return {};
	}

	//// 初始化参数
	int ret_idx = -1; // 是否有候选装甲板的标识
	bool is_small = false;  // 大小装甲板标识
//	double weight = 0;
	vector<candidate_target> candidate;  // 候选装甲板

	////////////////////////////// 筛选候选装甲板 ////////////////////////////////////
	for (size_t i = 0; i < match_rects.size(); i++){
		const RotatedRect & rect = match_rects[i].rect;

		// 如果矩形的宽高比小于阈值的话就是小装甲，否则是大装甲(初步判断)
		if (wh_ratio < _para.small_armor_wh_threshold)
			is_small = true;
		else
			is_small = false;

//        cout << "wh_ratio: " << wh_ratio << endl;
        //// 优先判断矩形角度
        // 现在这个旋转矩形的角度
        float cur_angle = rect.size.width > rect.size.height ? \
                    abs(rect.angle) : 90 - abs(rect.angle);
        // 目标如果太倾斜的话就筛除
        if (cur_angle > 28) continue; // (其实可以降到26)


    // TODO: 这个代码块可以用ID识别替代
    // TODO: 添加装甲板ID，时间戳属性

        {//// 用均值和方差去除中间太亮的图片（例如窗外的灯光等）
            RotatedRect screen_rect = RotatedRect(rect.center, Size2f(rect.size.width * 0.88, rect.size.height),
                                                  rect.angle);
            //// 将装甲板两边的灯条去除
            Size size = Point(src.cols, src.rows);
            Point p1, p2;
            int x = screen_rect.center.x - screen_rect.size.width / 2 + _dect_rect.x;
            int y = screen_rect.center.y - screen_rect.size.height / 2 + _dect_rect.y;
            //// 初步理解是为了再dect_rect里面处理，但是很奇怪///答：坐标轴的转换，前面是使用截取坐标（_src）后面用的是原图坐标
            p1 = Point(x, y - 40);
            x = screen_rect.center.x + screen_rect.size.width / 2 + _dect_rect.x;
            y = screen_rect.center.y + screen_rect.size.height / 2 + _dect_rect.y;
            p2 = Point(x, y);
            Rect roi_rect = Rect(p1, p2);
            Mat roi;
            if (makeRectSafe(roi_rect, size)) {
                roi = src(roi_rect).clone();
                Mat mean, stdDev;
                double avg, stddev;
                meanStdDev(roi, mean, stdDev);////mean、stdDev为R3向量，三个维度分别表示bgr三通道的均值和标准差
                avg = mean.ptr<double>(0)[0];
                stddev = stdDev.ptr<double>(0)[0];
#ifdef SHOW_DEBUG_IMG
                cout << "                                            " << avg << endl;
                cout << "                                            " << stddev << endl << endl;
    //            putText(roi, to_string(int(avg)), rects[i].center, CV_FONT_NORMAL, 1, Scalar(0, 255, 255), 2);
                imshow("2.jpg", roi);
#endif
                // 阈值可通过实际测量修改
                if (avg > 57.00)
                    continue;
                if (stddev < 15.8)
                    continue;
            }
        }



		// 现在这个旋转矩形的高度（用来判断远近）
		int cur_height = MIN(rect.size.width, rect.size.height);


        int cur_area = rect.size.width * rect.size.height;
        Point2i cur_center = {static_cast<int>(rect.center.x + _dect_rect.x), static_cast<int>(rect.center.y + _dect_rect.y)};
		// 把矩形的特征信息push到一个候选vector中
		candidate.push_back(candidate_target{cur_center, cur_height, cur_area, cur_angle, static_cast<int>(i), is_small, match_rects[i].lr_rate, match_rects[i].angle_abs});
		ret_idx = 1;
	}
#ifdef SHOW_ALL_CANDIDATE
    Point2f ps[4];
    for (auto & i : candidate){
        match_rects[i.index].rect.points(ps);
        for (int j = 0; j < 4; j++) {
            line(src, {static_cast<int>(ps[j].x+_dect_rect.x), static_cast<int>(ps[j].y + _dect_rect.y)},
                 {static_cast<int>(ps[(j + 1) % 4].x+_dect_rect.x), static_cast<int>(ps[(j + 1) % 4].y+_dect_rect.y)}, CV_RGB(255, 0, 0));
        }
    }
#endif
	//================================ 到这里才结束循环 =======================================
	int final_index = 0;
	if (candidate.size() > 1){
        // TODO: 1.可以按面积排序（已优化）
        //       2.验证排序是否有意义
		// 将候选矩形按照高度大小排序，选出最大的（距离最近）
		sort(candidate.begin(), candidate.end(),
			 [](candidate_target & target1, candidate_target & target2){
				 return target1.armor_area > target2.armor_area;
			 });

        // TODO：在得到装甲板ID后需要更严谨的击打逻辑
		/**
		 * 下面的几个temp值可以筛选出最终要击打的装甲板，我只用了一两个效果已经挺好的了
		 * 只是偶尔还会有误识别的情况，可以将这几个temp值都组合起来进行最终的判断
		 * */

		float temp_angle = candidate[0].armor_angle;
		float temp_lr_rate = candidate[0].armor_lr_rate;
		float temp_angle_abs = candidate[0].armor_angle_abs;
		float temp_weight = temp_angle + temp_lr_rate;
        int temp_area = candidate[0].armor_area;
        Rect last_rect = _res_last.boundingRect();
        //// 利用上一帧目标区域实现装甲板目标追踪
        if (last_rect.contains(candidate[0].armor_center)){
            final_index = 0;
            goto label;
        }

        for (int i = 1; i < candidate.size(); i++){
            double angle = match_rects[candidate[i].index].rect.angle;
            if (last_rect.contains(candidate[i].armor_center)){
                final_index = i;
                break;
            }

			if ( candidate[0].armor_height / candidate[i].armor_height < 1.1){
				if (candidate[i].armor_angle < temp_angle
					/*&& (candidate[i].bar_lr_rate */){
					temp_angle = candidate[i].armor_angle;
                    if (candidate[i].armor_lr_rate < 1.66 || candidate[i].armor_lr_rate > 0.6) {final_index = i; }
				}
			}
		}
	}

#ifdef SHOW_CANDIDATE_IMG
	Mat rect_show;
    _src.copyTo(rect_show);
//     候选区域
    Point2f vertices[4];
    match_rects[final_index].rect.points(vertices);
    putText(rect_show, to_string(int(match_rects[final_index].rect.angle)), match_rects[final_index].rect.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    for (int i = 0; i < 4; i++)
        line(rect_show, vertices[i], vertices[(i + 1) % 4], CV_RGB(255, 0, 0));
    imshow("final_rect", rect_show);
#endif
    label:
	// ret_idx为 -1 就说明没有目标
	if (ret_idx == -1){
		_is_lost = true;
		return {};
	}
	// 否则就证明找到了目标
	_is_lost = false;
	_is_small_armor = candidate[final_index].is_small_armor;
	RotatedRect ret_rect = match_rects[candidate[final_index].index].rect;
    _res_last_matched = match_rects[candidate[final_index].index];  // 前一帧的目标装甲板参数
	return ret_rect;
}




cv::RotatedRect ArmorDetector::boundingRRect(const cv::RotatedRect & left, const cv::RotatedRect & right){
    //TODO：这里可以改进
	// 将左右边的灯条拟合成一个目标旋转矩形，没有考虑角度
	const Point & p_left_center = left.center, & p_right_center = right.center;
	Point2f center = (p_left_center + p_right_center) / 2.0;

	// 短边为width，长边为height
	double width_left = MIN(left.size.width, left.size.height);
	double width_right = MIN(right.size.width, right.size.height);
	double height_left = MAX(left.size.width, left.size.height);
	double height_right = MAX(right.size.width, right.size.height);

	float width = POINT_DIST(p_left_center, p_right_center) - (width_left + width_right) / 2.0;  // 减去灯条宽
	float height = std::max(height_left, height_right);  // 高为最大值
	//float height = (wh_l.height + wh_r.height) / 2.0;  // 高为均值
    //TODO：这里需要测试角度
    float angle = std::atan2(right.center.y - left.center.y, right.center.x - left.center.x);
	return RotatedRect(center, Size2f(width, height), angle * 180 / CV_PI);
}

bool ArmorDetector::Contain(RotatedRect &match_rect, vector<RotatedRect> &Lights, size_t &i, size_t &j)
{
    Rect bound_rect = match_rect.boundingRect();
    //// 用灯条中心点, 顶点和底点筛选
    for (size_t k=i+1;k<j;)
    {
        Point2f light_ps_center;
        Point2f light_ps_top;
        Point2f light_ps_bottom;
        Point2f *light_ps;

        Lights[k].points(light_ps);
        light_ps_center = Lights[k].center;
        light_ps_top = (light_ps[0] + light_ps[1]) / 2;
        light_ps_bottom = (light_ps[2] + light_ps[3]) / 2;

        if (bound_rect.contains(light_ps_center) ||bound_rect.contains(light_ps_top) || bound_rect.contains(light_ps_bottom))
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return false;
}

//将之前的各个函数都包含在一个函数中，直接用这一个
cv::RotatedRect ArmorDetector::getTargetAera(const cv::Mat & src, const int & sb_mode, const int & jd_mode){
    // 传入参数为哨兵模式和吊射基地模式
    sentry_mode = sb_mode;
    base_mode = jd_mode;
    setImage(src);  // 设置当前帧截取ROI图像
    vector<matched_rect> match_rects = findTarget(); // 获取候选装甲板矩形
    RotatedRect final_rect = chooseTarget(match_rects, src);  // 选择合适目标

    if(final_rect.size.width != 0){
        // 加上偏移量，装甲板在获取原图坐标 （前面那些的坐标都是不对的，所以不能用前面那些函数解角度）
        final_rect.center.x += _dect_rect.x;
        final_rect.center.y += _dect_rect.y;
        _res_last = final_rect;
        _lost_cnt = 0;
    }
    else{
        _find_cnt = 0;
        ++_lost_cnt;
        // TODO：添加死亡缓存
        // TODO：丢帧扩大筛选可结合上一帧目标区域和前两帧目标中心坐标的位移差，进行ID识别，判断该区域是否存在黑色装甲板，保证识别的连续性
        // 逐次加大搜索范围（根据相机帧率调整参数）
        if (_lost_cnt < 8)
            _res_last.size = Size2f(_res_last.size.width, _res_last.size.height);
        else if(_lost_cnt == 9)
            _res_last.size =Size2f(_res_last.size.width * 1.1, _res_last.size.height * 1.6);
        else if(_lost_cnt == 12)
            _res_last.size = Size2f(_res_last.size.width * 1.5, _res_last.size.height * 1.5);
        else if(_lost_cnt == 15)
            _res_last.size = Size2f(_res_last.size.width * 1.2, _res_last.size.height * 1.2);
        else if (_lost_cnt == 18)
            _res_last.size = Size2f(_res_last.size.width * 1.2, _res_last.size.height * 1.2);
        else if (_lost_cnt > 33 )
            _res_last = RotatedRect();
    }
    //TODO：添加last_armors向量，储存前一帧检测到的装甲板合集
    //TODO：添加multimap类型的target_centers, 储存前50（待定）帧目标装甲板中心目标，以及ID
    return final_rect;
}