#include "camera.h"
#include "ArmorDetector.hpp"
#include "KAl.h"
#include <opencv2/core/cvstd.hpp>
#include "CRC_Check.h"
#include"serialport.h"
//#include<X11/Xlib.h>
#include"Thread.hpp"

//#define DETECT
#define PREDICT

using namespace cv;

pthread_t thread1;
pthread_t thread2;
pthread_t thread3;

pthread_mutex_t mutex_new;  // 互斥量
pthread_cond_t cond_new;
pthread_mutex_t mutex_kal;
pthread_cond_t cond_kal;

cv::Mat  quan_src;
float quan_ab_pitch;
float quan_ab_yaw;
float quan_ab_roll;
float quan_speed;


bool is_start = false;  // 识别标识
bool is_kal = false;  // 卡尔曼预测标识
           
Mat src;   

int main(void)
{
	XInitThreads();
	pthread_mutex_init(&mutex_new, NULL);  // 以动态方式创建互斥锁
	pthread_cond_init(&cond_new, NULL);  // 初始化一个条件变量

    // 创建线程，运行相关的线程函数
	pthread_create(&thread1, NULL, Build_Src, NULL);
	pthread_create(&thread2, NULL, Armor_Kal, NULL);
	pthread_create(&thread3, NULL, Kal_predict, NULL);

    // 等待线程执行结束
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
    // 注销一个互斥锁
	pthread_mutex_destroy(&mutex_new);
	return 0;
}
