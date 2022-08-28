#include "camera.h"
#include <iostream>



bool Camera::init()
{
	//初始化sdk相机
	CameraSdkInit(0);
	//获取连接到当前pc端的相机设备
	int camera_status = CameraEnumerateDevice(camera_list, &pid);
	if (camera_status != CAMERA_STATUS_SUCCESS)
	{
		std::cout << "CameraEnumerateDevice fail with" << camera_status << "!" << std::endl;
		return false;
	}
	
	//初始化相机设备并创建相机句柄
	if (CameraInit(camera_list, -1, -1, &h_camera) != CAMERA_STATUS_SUCCESS)
	{
		CameraUnInit(h_camera);
		return false;
	}
	
	//获取相机设备的特性结构体
	auto status = CameraGetCapability(h_camera, &capbility);
	
	//获取当前相机的帧率模式数量
	//std::cout << capbility.iFrameSpeedDesc << std::endl;
	
	
	if (status != CAMERA_STATUS_SUCCESS)
	{
		std:: cout << "get capbility failed" << std::endl;
		return false;
	}
	
	//创建rgb图像数据缓冲区
	rgb_buffer = (unsigned char*)malloc(capbility.sResolutionRange.iHeightMax *
										capbility.sResolutionRange.iWidthMax * 3
	);
	
	//设置为手动设置曝光时间，在2毫秒以下
	CameraSetAeState(h_camera, false);
	
	//降低曝光时间，一般为了防止图像拖影将曝光时间调为2毫秒以下，该函数以微秒为单位
	CameraSetExposureTime(h_camera, 1650.0);
	//CameraSetExposureTime(h_camera, 15000.0);
	
	//调高图像模拟增益值，使
	//模拟增益值最大为192
	CameraSetAnalogGain(h_camera, 192);
	
	//CameraSetFrameSpeed(h_camera, 3);
	
	//int g;
	//double t;
	//获取相机当前的曝光时间
	//CameraGetExposureTime(h_camera, &t);
	
	//CameraGetAnalogGain(h_camera, &g);
	//相机开始图像采集
	CameraPlay(h_camera);
	
	//std::cout << t / 1000.0 << std::endl;
	//std::cout << g << std::endl;
	
	//设置getimagebuffer的输出格式
	if (capbility.sIspCapacity.bMonoSensor)
	{
		channel = 1;
		CameraSetIspOutFormat(h_camera, CAMERA_MEDIA_TYPE_MONO8);
		
	}
	else
	{
		channel = 3;
		CameraSetIspOutFormat(h_camera, CAMERA_MEDIA_TYPE_BGR8);
		
	}
	return true;
}

bool Camera::read_frame_raw()
{
	//获取相机句柄
	if (CameraGetImageBuffer(h_camera, &frame_h, &pbybuffer, 100000) == CAMERA_STATUS_SUCCESS)
	{
		if (ipiimage)
		{
			//将图像数据头指针释放
			cvReleaseImageHeader(&ipiimage);
		}
		//创建raw图像数据的头指针
		ipiimage = cvCreateImageHeader(cvSize(frame_h.iWidth, frame_h.iHeight), IPL_DEPTH_8U, 1);
		
		//由于sdk读取数据是从底部到顶部读取，所以要将图像数据垂直翻转
		//CameraFlipFrameBuffer(pbybuffer, &frame_h, 3);
		
		//将原始图像数据缓冲区中的图像数据赋给头指针
		cvSetData(ipiimage, pbybuffer, frame_h.iWidth);
		
		
		
		return true;
		
	}
	
	else
	{
		return false;
	}
}

bool Camera::read_frame_rgb()
{
	//从相机句柄对应的相
	if (CameraGetImageBuffer(h_camera, &frame_h, &pbybuffer, 10000000) == CAMERA_STATUS_SUCCESS)
	{
		//将原始图像数据转为bgr图像数据储存到rgb_buffer中
		CameraImageProcess(h_camera, pbybuffer, rgb_buffer, &frame_h);
		if (ipiimage)
		{
			//释放掉图像数据的头指针，相当于释放全部数据
			cvReleaseImageHeader(&ipiimage);
		}
		
		//新建一个图像数据头指针
		ipiimage = cvCreateImageHeader(cvSize(frame_h.iWidth, frame_h.iHeight), IPL_DEPTH_8U, channel);
		
		//将图像数据垂直翻转
		//CameraFlipFrameBuffer(rgb_buffer, &frame_h, 3);
		
		//将rgb图像数据缓冲区里的图像数据给头指针指向
		cvSetData(ipiimage, rgb_buffer, frame_h.iWidth * channel);
		
		
		
		//delete rgb_buffer;
		return true;
	}
	else
	{
		
		return false;
	}
	
}


//bool Camera::transform_src_data(cv::Mat &src)
//{
//	//将IPIImage原始数据图像转化为opencv的Mat图像类型
//	src = cv::cvarrToMat(ipiimage).clone();
//	return true;
//}

bool Camera::release_data()
{
	//将图像缓冲区中的图像数据释放掉
	CameraReleaseImageBuffer(h_camera, pbybuffer);
	
	cvReleaseImageHeader(&ipiimage);
	return true;
}


bool Camera::record_start()
{
	if (record_state != RECORD_START)
	{
		int FormatList[] = {1,2,3,4,0};
		for (int i = 0; i < 5; i++)
		{
			//printf("a\n");
			int iret = CameraInitRecord(h_camera, FormatList[i],path, FALSE, 100,30);
			if (iret == CAMERA_STATUS_SUCCESS)
			{
				record_state = RECORD_START;
				printf("record start successfully!!!\n");
				return true;
				
			}
		}
	}
	return false;
}

bool Camera::camera_record()
{
	if (record_state == RECORD_START)
	{
		//图像头指针复制，第一个参数是复制到的地址，第二个参数是提供复制数据的指针
		memcpy(&record_hframe, &frame_h, sizeof(tSdkFrameHead));
		CameraFlipFrameBuffer(rgb_buffer, &record_hframe, 1);
		CameraPushFrame(h_camera, rgb_buffer, &record_hframe);
		return true;
	}
	
	
}

Camera::~Camera()
{
	if (record_state == RECORD_START)
	{
		CameraStopRecord(h_camera);
		printf("already stop record and record saved!!!\n");
		record_state = RECORD_STOP;
	}
	else
	{
		printf("No record is started!!!\n");
	}
	CameraUnInit(h_camera);
	if (rgb_buffer != nullptr)
		free(rgb_buffer);
}

