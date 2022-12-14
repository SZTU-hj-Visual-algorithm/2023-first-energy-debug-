#include "Thread.hpp"
#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>


using namespace cv;

bool is_continue = true; // 未读取到摄像头将设为false

typedef struct form
{
	RotatedRect ROT;
	int Armor_type;  // 装甲板类型
	float a[4];  // 机器人状态数据
	int is_get;  // 是否从串口读取到数据
	int mode;  // 识别模式（小符、大符、装甲板）
	int buff_target_is_get;  // 是否获取打符目标
}form;

form send_data;  // 装甲板识别线程发送给其他进程的数据

Mat ka_src_get;

SerialPort port("/dev/ttyUSB0");

void* Build_Src(void* PARAM)
{
	Mat get_src;
	auto camera_warper = new Camera;
	printf("camera_open\n");
//	int count = 0;
	if (camera_warper->init())
	{
		while (is_continue && !(waitKey(10) == 27))
		{
//			string str = "./en/"+ to_string(count) + ".jpg";
			if (camera_warper->read_frame_rgb())
			{
				//printf("1\n");
				get_src = cv::cvarrToMat(camera_warper->ipiimage).clone();
				pthread_mutex_lock(&mutex_new);  // 互斥锁上锁
				{
					get_src.copyTo(src);
//					imwrite(str,src);
					is_start = true;  // 读取图片，进入开始状态
					pthread_cond_signal(&cond_new);  // 发送一个信号给另外一个正在处于阻塞等待状态的线程(无条件等待),使其脱离阻塞状态,继续执行
					pthread_mutex_unlock(&mutex_new);  // 互斥锁解锁
					imshow("src",src);

					camera_warper->release_data();
				}
				//camera_warper->record_start();
				//camera_warper->camera_record();
			}
			else
			{
				src = cv::Mat();
			}
//			count++;
		}
		camera_warper->~Camera();
		is_continue = false;
	}
	else
	{
		printf("No camera!!\n");
		is_continue = false;
	}
}

void* Armor_Kal(void* PARAM)
{
	ArmorDetector target_detector = ArmorDetector();
//	shibie.enermy_color = RED;
//	RotatedRect mubiao;
	Mat src_copy;
	long int time_count = 0;  // 系统时间
	energy_pre E_predicter;  // 预测类

	port.initSerialPort();  // 初始化串口
	
	sleep(2);
	printf("Armor_open\n");
	while (is_continue)
	{
		pthread_mutex_lock(&mutex_new);  // 锁住mutex_new对象
        // 如果没有读取到图片
		while (!is_start) {

			pthread_cond_wait(&cond_new, &mutex_new); // 无条件等待，待读取图片后唤醒

		}

		is_start = false;  // 开始进程后初始化变量

		src.copyTo(src_copy);

		//imshow("src_copy",src_copy);

		pthread_mutex_unlock(&mutex_new);  // 互斥锁解锁
		float lin[4];
		int mode_temp/* = 0x22*/;  //模式
//		lin[0] = 0.0;  // pitch
// 		lin[1] = 0.0;  // yaw
//		lin[2] = 0.0;  // roll
//		lin[3] = 0.0;  // ball_speed
		bool small_energy = false;
		int lin_is_get;
		//lin_is_get = true;
		lin_is_get = port.get_Mode1(mode_temp, lin[0], lin[1], lin[2], lin[3],target_detector.enermy_color);  // 通过串口读取数据
        /// 调试手动设置模式和识别颜色
//		mode_temp = 0x22;
//		shibie.enermy_color=RED;

		//printf("mode:%x\n",shibie.enermy_color);
		//printf("speed:%lf\n",lin[3]);

		if (mode_temp == 0x21)
		{


			RotatedRect target_get = target_detector.getTargetAera(src_copy, 0, 0);  // 获取装甲板目标

			pthread_mutex_lock(&mutex_kal);  // 锁定进程
			send_data.ROT = target_get;
			send_data.Armor_type = target_detector.isSamllArmor();  // 是否为小装甲板

            // 状态数据
			send_data.a[0] = lin[0];
			send_data.a[1] = lin[1];
			send_data.a[2] = lin[2];
			send_data.a[3] = lin[3];
			src_copy.copyTo(ka_src_get); 
			send_data.mode = mode_temp;
			send_data.is_get = lin_is_get;

			is_kal = true;
			pthread_cond_signal(&cond_kal);  // 唤醒预测线程
			pthread_mutex_unlock(&mutex_kal);  // 解锁线程
		}
		else if (mode_temp == 0x22)
		{
		    small_energy = false;
			//printf("big energy!!\n");
			ka_src_get.copyTo(quan_src);
			quan_ab_pitch = lin[0];
			quan_ab_yaw = lin[1];
			E_predicter.ab_roll = lin[2];
			E_predicter.SPEED = lin[3];
			send_data.is_get = lin_is_get;
			time_count = getTickCount(); // 获取系统时间
			//printf("quan_pitch:%f",quan_ab_pitch);
			//printf("quan_yaw:%f",quan_ab_yaw);
			if (E_predicter.energy_detect(src, target_detector.enermy_color))
			{
				if (E_predicter.energy_predict_aim(time_count,small_energy))
				{
				    pthread_mutex_lock(&mutex_kal);
				    send_data.a[0] = E_predicter.E_pitch - quan_ab_pitch;
				    send_data.a[1] = E_predicter.E_yaw - quan_ab_yaw;
				    printf("de_yaw:%f     de_pitch:%f\n",-send_data.a[1],-send_data.a[0]);
				    send_data.mode = mode_temp;
				    send_data.buff_target_is_get = 0x31;
                    is_kal = true;
				    pthread_cond_signal(&cond_kal);
				    pthread_mutex_unlock(&mutex_kal);
				}
				else
				{
				    pthread_mutex_lock(&mutex_kal);
				    send_data.buff_target_is_get = 0x32;
                    is_kal = true;
				    pthread_cond_signal(&cond_kal);
				    pthread_mutex_unlock(&mutex_kal);
				}
			}
			else
			{
				//imshow("bad",src);
				pthread_mutex_lock(&mutex_kal);
				send_data.buff_target_is_get = 0x32;
                is_kal = true;
				pthread_cond_signal(&cond_kal);
				pthread_mutex_unlock(&mutex_kal);
			}
		}
		else if (mode_temp == 0x23)
		{
			small_energy = true;
			//printf("samll energy!!\n");
			ka_src_get.copyTo(quan_src);
			quan_ab_pitch = lin[0];
			quan_ab_yaw = lin[1];
			E_predicter.ab_roll = lin[2];
			E_predicter.SPEED = lin[3];
			send_data.is_get = lin_is_get;
			time_count = getTickCount();  // 获取系统时间
			//printf("quan_pitch:%f",quan_ab_pitch);
			//printf("quan_yaw:%f",quan_ab_yaw);
			if (E_predicter.energy_detect(src, target_detector.enermy_color)) //
			{
				if (E_predicter.energy_predict_aim(time_count,small_energy))
				{
					pthread_mutex_lock(&mutex_kal);
					send_data.a[0] = E_predicter.E_pitch - quan_ab_pitch;
					send_data.a[1] = E_predicter.E_yaw - quan_ab_yaw;
					send_data.mode = mode_temp;
					send_data.buff_target_is_get = 0x31;
                    is_kal = true;
					pthread_cond_signal(&cond_kal);
					pthread_mutex_unlock(&mutex_kal);
				}
				else
				{
					pthread_mutex_lock(&mutex_kal);
					send_data.buff_target_is_get = 0x32;
                    is_kal = true;
					pthread_cond_signal(&cond_kal);
					pthread_mutex_unlock(&mutex_kal);
				}

			}
			else
			{
				pthread_mutex_lock(&mutex_kal);
				send_data.buff_target_is_get = 0x32;
                is_kal = true;
				pthread_cond_signal(&cond_kal);
				pthread_mutex_unlock(&mutex_kal);
			}
		}
	}
}

void* Kal_predict(void* PARAM)
{
	VisionData vdata;
	KAL ka;  // 预测类
	kal_filter kf;  // 卡尔曼滤波类
	double time_count = 0;  // //获取系统启动后的毫秒数
	//SerialPort port("/dev/ttyUSB"); //d
	//port.initSerialPort(); //d
	kf = ka.init();  // 初始化卡尔曼滤波
	form get_data;
	sleep(3);
	printf("kal_open\n");
    // 识别返回数据变量定义
	int is_get;
	int mode;
	int is_send;
	float ji_pitch,ji_yaw;
	int pan_wu = 0;

	RotatedRect target;

	while (is_continue)
	{
		pthread_mutex_lock(&mutex_kal);  // 进程上锁

		while (!is_kal) {

			pthread_cond_wait(&cond_kal, &mutex_kal);  // 无条件等待，待识别装甲板完成后唤醒

		}

        is_kal = false;  // 进入预测进程，初始化标识
	    /// ??? ka_src_get原始数据从哪来的 ？？？
		ka_src_get.copyTo(ka._src);
		ka_src_get.copyTo(quan_src);  // (未使用)

        // 获取装甲板进程数据
		ka.ab_pitch = send_data.a[0];
		ka.ab_yaw = send_data.a[1];
		ka.ab_roll = send_data.a[2];
		ka.SPEED = send_data.a[3];
		is_get = send_data.is_get;
		mode = send_data.mode;
		is_send = send_data.buff_target_is_get;
		pthread_mutex_unlock(&mutex_kal);  // 进程解锁
        target=send_data.ROT;
		ka.type = (send_data.Armor_type) ? 1 : 2;

		if(is_get)
		{
			if (mode == 0x21)
			{
				if (ka.predict(target, kf, time_count))
				{
					time_count = (double)getTickCount();
					ji_pitch = ka.send.pitch;
					ji_yaw = ka.send.yaw;
					vdata = { -ji_pitch, -ji_yaw, 0x31 };  // 云台转动数据
					printf("yaw:%f\npitch:%f\n", -ka.send.yaw, -ka.send.pitch);
					port.TransformData(vdata);
					port.send();
					pan_wu = 0;
				}
				else
				{
					if(pan_wu<=10)
					{
						
						vdata = { -ji_pitch, -ji_yaw, 0x31 };
						printf("yaw:%f\npitch:%f\npan_wu:%d\n",-ji_yaw, -ji_pitch,pan_wu);
						port.TransformData(vdata);
						port.send();
						pan_wu++;
					}else
					{
						ji_yaw = 0.0 - ka.ab_yaw;
						ji_pitch = 0.0 - ka.ab_pitch;
						ka.sp_reset(kf);
						vdata = { -ji_pitch, -ji_yaw, 0x32 };
						//printf("real none!!");
						//printf("chong\n");
						port.TransformData(vdata);
						port.send();
					}
				}
			}
			else if ((mode == 0x22)||(mode == 0x23))
			{
				
				if(is_send == 0x31)
				{
					//printf("dafu_yaw:%f\ndafu_pitch:%f\n", -ka.ab_yaw, -ka.ab_pitch);
					printf("fa_yaw:%f     fa_pitch:%f\n",-ka.ab_yaw,-ka.ab_pitch);
					vdata = { -ka.ab_pitch, -ka.ab_yaw, 0x31 };
					pan_wu = 0;
				}				
				else if (is_send == 0x32)
				{
					if (pan_wu <= 6)
					{
						printf("dafu_yaw:%f\ndafu_pitch:%f\n", -ka.ab_yaw, -ka.ab_pitch);
						vdata = { -ka.ab_pitch, -ka.ab_yaw, 0x31 };
						pan_wu++;
					}
					else
					{
						vdata = { 0.0f, 0.0f, 0x32};
						
					}
				}				
				port.TransformData(vdata);
				port.send();
			}
		}
	}
}

