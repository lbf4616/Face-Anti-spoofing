#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include <vector>
#include <stdlib.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
 
using namespace dlib;  
using namespace std;  
using namespace Eigen;
using namespace cv;
using namespace cv::ml;

std::vector<double> rfft(std::vector<double> x);
std::vector<double> detrend(std::vector<double> x);
double max(std::vector<double> x);
double sum(std::vector<double> x);
std::vector<double> LBP(cv::Mat face, int P, int R);
 
//double dur;
//clock_t start, end, end1, end2, end3, end4, end5;


int main()
{
	Ptr<SVM> svmLBP = StatModel::load<SVM>(".\\LBP.txt");
	Ptr<SVM> svmHR = StatModel::load<SVM>(".\\3DMAD.txt");
	// Load face detection and pose estimation models.  
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	std::vector<double> B, G, R;
	cv::Mat t_mat(2, 3, CV_32FC1);
	int ans = 0;
	//Grab and process frames until the main window is closed by the user.  
	/*for (int i = 1; i <36; i++)
	{	
		for (int j = 1; j < 3; j++)
		{
			clock_t start, end;
			start = clock();
			string J = to_string(j);
			string VideoNum;
			string I = to_string(i);
			if (i < 10)
			{
				VideoNum = "0" + I + "_02_0" + J + ".mp4";
			}
			else 
			{
				VideoNum = I + "_02_0" + J + ".mp4";
			}*/
			//string dir = "D:\\Anti-spoofing\\databases for anti-spoofing\\3DMAD-spoofing\\videos\\03\\"+VideoNum;
			//string dir = "D:\\Anti-spoofing\\databases for anti-spoofing\\MSU\\converted videos\\realvideo\\" + VideoNum;
			//std::vector<double> B, G, R;
	try
	{
		cv::VideoCapture cap(0); //capture video
		//int nFrame = cap.get(CV_CAP_PROP_FRAME_COUNT);//Total frames
		int times = 0; //Every three frames extract LBP features
		int t = -1; //Only extract HRfeature once
		int x01 = 0, x15 = 0, x03 = 0, x13 = 0, x05 = 0, y05 = 0, y03 = 0, y29 = 0, x09 = 0, x07 = 0, x10 = 0, x06 = 0, x11 = 0, x04 = 0, x12 = 0, y07 = 0, y06 = 0, y04 = 0, x39=0, y39=0, x42=0, y42=0;
		int B1 = 0, G1 = 0, R1 = 0;
		float Tpix1 = 0.0, Tpix2 = 0.0, Tpix3 = 0.0, Tpix4 = 0.0, Tpix5 = 0.0;
		image_window win;
		while (1)
		{
			t += 1;
			cout << t << endl;
			cv::Mat temp;
			cap.read(temp);
			
			cv_image<bgr_pixel>dlib_image(temp);

			std::vector<dlib::rectangle> faces = detector(dlib_image);
			// Find the pose of each face. 
			//dlib::rectangle dets = dlib::detector(cimg);
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(dlib_image, faces[i]));
			//temp = temp2.clone();
			if (!shapes.empty())
			{
				//HRfeature extracting, Get ROI
				x01 = shapes[0].part(1).x() + 5;
				x15 = shapes[0].part(15).x() - 5;
				x03 = shapes[0].part(3).x() + 5;
				x13 = shapes[0].part(13).x() - 5;
				x05 = shapes[0].part(5).x() + 5;
				y05 = shapes[0].part(5).y();
				y03 = shapes[0].part(3).y();
				y29 = shapes[0].part(1).y();
				x09 = shapes[0].part(9).x() + 5;
				x07 = shapes[0].part(7).x() - 5;
				x10 = shapes[0].part(10).x() - 5;
				x06 = shapes[0].part(6).x() + 5;
				x11 = shapes[0].part(11).x() - 5;
				x04 = shapes[0].part(4).x() + 5;
				x12 = shapes[0].part(12).x() - 5;
				y07 = shapes[0].part(7).y();
				y06 = shapes[0].part(6).y();
				y04 = shapes[0].part(4).y();

				B1 = 0, G1 = 0, R1 = 0;
				int ROI1L = x09 - x07;     //set ROI1
				int ROI1H = y07 - y06;
				Tpix1 = ROI1L * ROI1H;
				int ROI2L = x10 - x06;     //2
				int ROI2H = y06 - y05;
				Tpix2 = ROI2L * ROI2H;
				int ROI3L = x11 - x05;     //3
				int ROI3H = y05 - y04;
				Tpix3 = ROI3L * ROI3H;
				int ROI4L = x12 - x04;     //4
				int ROI4H = y04 - y03;
				Tpix4 = ROI4L * ROI4H;
				int ROI5L = x13 - x03;     //5
				int ROI5H = y03 - y29;
				Tpix5 = ROI5L * ROI5H;

				// Calculate the ROI1 Blue
				for (int j = y06; j < y07 + 1; j++)
				{
					for (int i = x07; i < x09 + 1; i++)
					{
						B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
					}
				}
				// Calculate the ROI2 Blue
				for (int j = y05; j < y06 + 1; j++)
				{
					for (int i = x06; i < x10 + 1; i++)
					{
						B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
					}
				}
				// Calculate the ROI3 Blue
				for (int j = y04; j < y05 + 1; j++)
				{
					for (int i = x05; i < x11 + 1; i++)
					{
						B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
					}
				}
				// Calculate the ROI4 Blue
				for (int j = y03; j < y04 + 1; j++)
				{
					for (int i = x04; i < x12 + 1; i++)
					{
						B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
					}
				}
				// Calculate the ROI5 Blue
				for (int j = y29; j < y03 + 1; j++)
				{
					for (int i = x03; i < x13 + 1; i++)
					{
						B1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 0];
					}
				}


				// Calculate the ROI1 Green
				for (int j = y06; j < y07 + 1; j++)
				{
					for (int i = x07; i < x09 + 1; i++)
					{
						G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
					}
				}
				// Calculate the ROI2 Green
				for (int j = y05; j < y06 + 1; j++)
				{
					for (int i = x06; i < x10 + 1; i++)
					{
						G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
					}
				}
				// Calculate the ROI3 Green
				for (int j = y04; j < y05 + 1; j++)
				{
					for (int i = x05; i < x11 + 1; i++)
					{
						G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
					}
				}
				// Calculate the ROI4 Green
				for (int j = y03; j < y04 + 1; j++)
				{
					for (int i = x04; i < x12 + 1; i++)
					{
						G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
					}
				}
				// Calculate the ROI5 Green
				for (int j = y29; j < y03 + 1; j++)
				{
					for (int i = x03; i < x13 + 1; i++)
					{
						G1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 1];
					}
				}


				// Calculate the ROI1 Red
				for (int j = y06; j < y07 + 1; j++)
				{
					for (int i = x07; i < x09 + 1; i++)
					{
						R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
					}
				}
				// Calculate the ROI2 Red
				for (int j = y05; j < y06 + 1; j++)
				{
					for (int i = x06; i < x10 + 1; i++)
					{
						R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
					}
				}
				// Calculate the ROI3 Red
				for (int j = y04; j < y05 + 1; j++)
				{
					for (int i = x05; i < x11 + 1; i++)
					{
						R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
					}
				}
				// Calculate the ROI4 Red
				for (int j = y03; j < y04 + 1; j++)
				{
					for (int i = x04; i < x12 + 1; i++)
					{
						R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
					}
				}
				// Calculate the ROI5 Red
				for (int j = y29; j < y03 + 1; j++)
				{
					for (int i = x03; i < x13 + 1; i++)
					{
						R1 += temp.data[j * temp.cols * temp.elemSize() + i * temp.elemSize() + 2];
					}
				}
				double avgB = B1 / (Tpix1 + Tpix2 + Tpix3 + Tpix4 + Tpix5);  // All points get average
				double avgG = G1 / (Tpix1 + Tpix2 + Tpix3 + Tpix4 + Tpix5);
			    double avgR = R1 / (Tpix1 + Tpix2 + Tpix3 + Tpix4 + Tpix5);

				B.push_back(avgB);
				G.push_back(avgG);
				R.push_back(avgR);

				//Eyes inner corner coordinates;
				int x39 = shapes[0].part(39).x();
				int x42 = shapes[0].part(42).x();
				int y39 = shapes[0].part(39).y();
				int y42 = shapes[0].part(42).y();
				//The distance between two eyes(inner corner)
				double eyedis = sqrt((x42 - x39)*(x42 - x39) + (y42 - y39)*(y42 - y39));
				if (B.size() == 200)
				{
					std::vector<double> Yr, Yg, Yb;
					std::vector<double> Xr, Xg, Xb;
					std::vector<double> HRFeature;
					Yb = detrend(B);
					Yg = detrend(G);
					Yr = detrend(R);
					Xb = rfft(Yb);
					Xg = rfft(Yg);
					Xr = rfft(Yr);
					double Er, Eg, Eb, Tr, Tg, Tb;
					//cout << Xb.size() << " " << Xg.size() << " " << Xr.size() << " " << endl;
					Er = max(Xr);
					Eg = max(Xg);
					Eb = max(Xb);
					Tr = Er / sum(Xr);
					Tg = Eg / sum(Xg);
					Tb = Eb / sum(Xb);
					B.clear();
					G.clear();
					R.clear();

					Mat DataMat;
					DataMat = Mat::zeros(1, 6, CV_32FC1);
					DataMat.at<float>(0, 0) = Er;
					DataMat.at<float>(0, 1) = Eg;
					DataMat.at<float>(0, 2) = Eb;
					DataMat.at<float>(0, 3) = Tr;
					DataMat.at<float>(0, 4) = Tg;
					DataMat.at<float>(0, 5) = Tb;
					cout << Er << " " << Eg << " " << Eb << " " << Tr << " " << Tg << " " << Tb << endl;
					int res = 0;
					res = svmHR->predict(DataMat);
					cout << res << endl;

					//The index of zooming
					double index = 50 / eyedis;

					t_mat = cv::Mat::zeros(2, 3, CV_32FC1);
					cv::Mat dst;
					cv::Mat dst2;
					cv::Size src_sz1 = temp.size();
					cv::Size src_sz(src_sz1.width * index, src_sz1.height * index);

					t_mat.at<float>(0, 0) = index;
					t_mat.at<float>(1, 1) = index;
					t_mat.at<float>(0, 1) = 0.0;
					t_mat.at<float>(1, 0) = 0.0;
					t_mat.at<float>(0, 2) = 0.0;
					t_mat.at<float>(1, 2) = 0.0;

					//根据缩放矩阵进行仿射变换
					cv::warpAffine(temp, dst, t_mat, src_sz);
					//把眼睛放在固定点 剪切脸部
					cv::Rect rect(x39 * index - 37, y39 * index - 37, 128, 128);
					//The angle of rotation
					double angle = 180 / 3.14159 * atan((y42 - y39) / 2. / ((x42 - x39) / 2.));

					cv::Size dst_sz(src_sz.width, src_sz.height);
					//指定旋转中心
					cv::Point2f center(x39*index, y39*index);
					//获取旋转矩阵（2x3矩阵）
					cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
					//根据旋转矩阵进行仿射变换
					cv::warpAffine(dst, dst2, rot_mat, dst_sz);
					//从变换后的图片中按照rect进行切割
					cv::Mat image_cut = dst2(rect);      

					//cv::imwrite("d://lbf.jpg", image_cut);
					std::vector<double> lbp1, lbp2, lbp3, lbp4;

					lbp1 = LBP(image_cut, 8, 1);
					Mat DataMat2;
					DataMat2 = Mat::zeros(1, 177, CV_32FC1);
					int res2 = 0;
					Mat M2 = Mat(1, 177, CV_32FC1, (double*)lbp1.data());
					res2 = svmLBP->predict(M2);
					cout << res2 << endl;
					if (res == 1 && res2 == 1)
						ans = 1;
					else ans = 2;

					
					/*cv::line(temp, cv::Point(x39 - 2 * eyedis, y39 - 2 * eyedis), cv::Point(x39 - 2 * eyedis + 5 * eyedis, y39 - 2 * eyedis), cv::Scalar(0, 0, 255));
					cv::line(temp, cv::Point(x39 - 2 * eyedis + 5 * eyedis, y39 - 2 * eyedis), cv::Point(x39 - 2 * eyedis + 5 * eyedis, y39 - 2 * eyedis + 5 * eyedis), cv::Scalar(0, 0, 255));
					cv::line(temp, cv::Point(x39 - 2 * eyedis, y39 - 2 * eyedis), cv::Point(x39 - 2 * eyedis, y39 - 2 * eyedis + 5 * eyedis), cv::Scalar(0, 0, 255));
					cv::line(temp, cv::Point(x39 - 2 * eyedis, y39 - 2 * eyedis + 5 * eyedis), cv::Point(x39 - 2 * eyedis + 5 * eyedis, y39 - 2 * eyedis + 5 * eyedis), cv::Scalar(0, 0, 255));*/
				}
				if (ans == 1)
				{
					//cv::putText(temp, "Fake!", cv::Point(x39 - 0.5*eyedis, y39 - 3*eyedis), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(10, 10, 255), 2);
					cv::putText(temp, "Real!", cv::Point(x39 - 0.5*eyedis, y39 - 3 * eyedis), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(10, 255, 10), 2);
				}
				else if (ans == 2)
				{
					cv::putText(temp, "Fake!", cv::Point(x39 - 0.5*eyedis, y39 - 3 * eyedis), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(10, 10, 255), 2);
				}
				else cv::putText(temp, "Detecting...", cv::Point(x39 - 0.5*eyedis, y39 - 3 * eyedis), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 10, 10), 2);
		}
				
				//faces.clear();
				win.clear_overlay();
				win.set_image(dlib_image);
				//win.add_overlay(render_face_detections(shapes));
				//const dlib::rectangle r1(10,12,12,20);
				//const dlib::rectangle r2(10,14,10,14);
				//const dlib::rectangle r3(10, 16,10, 16);
				//const dlib::rectangle r4(10, 18, 10, 18);
				//const dlib::rectangle r5(10, 20, 10, 20);
				//win.add_overlay(r1);
				///*win.add_overlay(r2);
				/*win.add_overlay(r3);
				win.add_overlay(r4);
				win.add_overlay(r5);*/
				/*win.add_overlay(dlib::image_window::overlay_rect(r1, rgb_pixel(255, 0, 0), "*****  * *** *   * *****"));
				win.add_overlay(dlib::image_window::overlay_rect(r2, rgb_pixel(255, 0, 0), "   *     * *   *    * *    "));
				win.add_overlay(dlib::image_window::overlay_rect(r3, rgb_pixel(255, 0, 0), "  *    **    *   * **** "));
				win.add_overlay(dlib::image_window::overlay_rect(r4, rgb_pixel(255, 0, 0), "  *    *     *   * *    "));
				win.add_overlay(dlib::image_window::overlay_rect(r5, rgb_pixel(255, 0, 0), "  *    *     ***** *****"));*/
				
				win.add_overlay(faces);
				//imshow("window", temp);
				
				//waitKey(15);
				//temp.release();

			}
		}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
	system("pause");
}
	

std::vector<double> detrend(std::vector<double> x)
{
	int lambda = 50;
	int BL = x.size();
	MatrixXd Color(1, BL);
	for (int i = 0; i < BL; i++)
	{
		Color(0, i) = x[i];
	}
	int L = Color.cols();
	int m = L - 2, n = L;
	SparseMatrix<double> D2(m, n);
	D2.reserve(VectorXi::Constant(n, 3));
	for (int i = 0; i < m; ++i)
	{
		D2.insert(i, i) = 1;
		D2.insert(i, i + 1) = -2;
		D2.insert(i, i + 2) = 1;
	}
	D2.makeCompressed();
	SparseMatrix<double>  I(L, L);
	//Sparse eye I
	for (int i = 0; i < L; ++i)
	{
		I.insert(i, i) = 1;
	}
	SparseMatrix<double> ColorD1(L, L);
	ColorD1 = (I + (pow(lambda, 2) * (D2.adjoint()*D2)));
	//Sparse Matrix to Matrix
	MatrixXd ColorD2 = ColorD1;
	MatrixXd I2 = I;
	ColorD2 = (I2 - ColorD2.inverse())* (Color.adjoint());
	MatrixXd ColorD3 = ColorD2.adjoint();

	//Moving average filter
	int N = 5;
	MatrixXd sa(1, L);
	sa.setZero(1, L);
	std::vector<double> sla;
	double a;

	// Matrix to Vector, while set N-1 zeros at the begining of the vector
	for (int i = 0; i < N - 1; i++)
	{
		sla.push_back(0);
	}
	for (int i = 0; i < L; i++)
	{
		sla.push_back(ColorD3(0, i));
	}

	//Filter
	for (int i = 1; i <= L; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a += sla[i + j - 1] / N;
		}
		sa(0, i - 1) = a;
		a = 0;
	}

	//FIR Hd
	int n1 = 127, n2; //n orders of the filter
	double fln = 0.7, fhn = 4; //下届，上界
	double s, pi, wc1, wc2, delay, h[127], fs = 30;
	pi = 4.0*atan(1.0);//pi=PI;
	n2 = n1 / 2;
	delay = n1 / 2.0;
	wc1 = 2.0*pi*fln;
	wc2 = 2.0*pi*fhn;
	for (int i = 0; i <= n2; i++)
	{
		s = i - delay;
		h[i] = ((sin(wc2*s / fs) - sin(wc1*s / fs)) / (pi*s))* (0.54 - 0.46*cos(2 * i*pi / (n1 - 1)));//Bandpass with Hamming window	
		h[n1 - i] = h[i];
	}
	std::vector<double> Y;
	double y;
	for (int j = 0; j < L; j++)
	{
		for (int i = 0; i < n1; i++)
		{
			if (j >= i)
				y += h[i] * sa(0, j - i);
		}
		Y.push_back(y);
	}
	return Y;
}

//快速傅里叶变换
std::vector<double> rfft(std::vector<double> x)
{
	//std::vector<double> x = {1,2,3,4,5};
	int xz1 = x.size();
	int xz = x.size(); //必须是2的整数次幂
	int n;
	char s[20];
	itoa(xz, s, 2);//转为二进制
	string y = s;  //转为字符串
	int L = y.length();
	for (int i = 1; i < L; i++) //除第一位以外有1的
	{
		if (y[i] == '1')
		{
			n = pow(2, L);  //Next higher power of 2
			for (int j = xz; xz < n; xz++)//扩展后的x的后面用0补足
			{
				x.push_back(0);
			}
			break;
		}
	}

	int m, k, n1, n2, n4,i1, i2, i3, i4;
	double a, e, cc, ss, xt, t1, t2;
	
	for (int j = 1, i = 1; i < 16; i++)
	{
		m = i;
		j = 2 * j;
		if (j == n)break;
	}
	n1 = n - 1;
	for (int j = 0, i = 0; i < n1; i++)
	{
		if (i < j)
		{
			xt = x[j];
			x[j] = x[i];
			x[i] = xt;
		}
		k = n / 2;
		while (k < (j + 1))
		{
			j = j - k;
			k = k / 2;
		}
		j = j + k;
	}
	for (int i = 0; i < n; i += 2)
	{
		xt = x[i];
		x[i] = xt + x[i + 1];
		x[i + 1] = xt - x[i + 1];
	}
	n2 = 1;
	for (int k = 2; k <= m; k++)
	{
		n4 = n2;
		n2 = 2 * n4;
		n1 = 2 * n2;
		e = 6.2831850718 / n1;
		for (int i = 0; i < n; i += n1)
		{
			xt = x[i];
			x[i] = xt + x[i + n2];
			x[i + n2] = xt - x[i + n2];
			x[i + n2 + n4] = -x[i + n2 + n4];
			a = e;
			for (int j = 1; j <= (n4 - 1); j++)
			{
				i1 = i + j;
				i2 = i - j + n2;
				i3 = i + j + n2;
				i4 = i - j + n1;
				cc = cos(a);
				ss = sin(a);
				a = a + e;
				t1 = cc*x[i3] + ss*x[i4];
				t2 = ss*x[i3] - cc*x[i4];
				x[i4] = x[i2] - t2;
				x[i3] = -x[i2] - t2;
				x[i2] = x[i1] - t1;
				x[i1] = x[i1] + t1;
			}
		}
	}
	std::vector<double> p;
	p.push_back(2 * abs(x[0])/xz1);
	for (int i = 1; i < n / 2; i++)
	{
		p.push_back(2 * sqrt(pow(x[i], 2) + pow(x[n - i], 2))/xz1);
	}
	p.push_back(2 * abs(x[n / 2])/xz1);
	return p;
}

//Find max frequency
double max(std::vector<double> x)
{
	int Xsize = x.size();
	double max1 = 0.0;
	cout << Xsize << endl;
	for (int i = 12; i < 70; i++)//频率在13~70之间
	{
		if (x[i] > max1)
		{
			max1 = x[i];
		}
	}
	return max1;
}

//Calculate the sum of frequency
double sum(std::vector<double> x)
{
	int Xsize = x.size();
	double sum1 = 0.0;
	for (int i = 12; i < 70; i++)
	{
			sum1 += x[i];
	}
	return sum1;
}

//LBP with uniform pattern 59 dimensional vector
std::vector<double> LBP(cv::Mat face, int P, int R)
{
	std::vector<double> histogram;
	std::vector<int> feature;
	std::vector<int> feature1;
	std::vector<int> feature2;
	double num = 16129.0;
	// Split the image into different channels
	int LBP[59] = { 0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255 };	
	//Extract LBP features	Green
	for (int i = R - 1; i < face.cols - R; i++)
	{	

		for (int j = R - 1; j < face.rows - R; j++)
		{
			int neighbor[] = { face.at<cv::Vec3b>(i - R, j - R)[0], face.at<cv::Vec3b>(i, j - R)[0], face.at<cv::Vec3b>(i + R, j - R)[0], face.at<cv::Vec3b>(i + R, j)[0], +face.at<cv::Vec3b>(i + R, j + R)[0], face.at<cv::Vec3b>(i, j + R)[0], face.at<cv::Vec3b>(i - R, j + R)[0], face.at<cv::Vec3b>(i - R, j)[0] };
			int value = 0;
			for (int k = 0; k < 8; k++)
			{
				if (face.at<cv::Vec3b>(i, j)[0] > neighbor[k])
				{
					value += pow(double(2), k);				
				}
			}
			//img.at<uchar>(j, i) = value;
			feature.push_back(value);			
		}		
	}
	for (int i1 = 0; i1 < 59; i1++)
	{
		double x = count(feature.begin(), feature.end(), LBP[i1]) / num;
		histogram.push_back(x);
	}

	//Extract LBP features	Blue
	for (int i = R - 1; i < face.cols - R; i++)
	{

		for (int j = R - 1; j < face.rows - R; j++)
		{
			int neighbor[] = { face.at<cv::Vec3b>(i - R, j - R)[1], face.at<cv::Vec3b>(i, j - R)[1], face.at<cv::Vec3b>(i + R, j - R)[1], face.at<cv::Vec3b>(i + R, j)[1], +face.at<cv::Vec3b>(i + R, j + R)[1], face.at<cv::Vec3b>(i, j + R)[1], face.at<cv::Vec3b>(i - R, j + R)[1], face.at<cv::Vec3b>(i - R, j)[1] };
			int value = 0;
			for (int k = 0; k < 8; k++)
			{
				if (face.at<cv::Vec3b>(i, j)[1] > neighbor[k])
				{
					value += pow(double(2), k);
				}
			}
			feature1.push_back(value);
		}
	}

	//Count the histogram
	for (int i = 0; i < 59; i++)
	{
		double x = count(feature1.begin(), feature1.end(), LBP[i]) / num;
		histogram.push_back(x);
	}

	//Extract LBP features	Red
	for (int i = R - 1; i < face.cols - R; i++)
	{

		for (int j = R-1; j < face.rows - R; j++)
		{
			int neighbor[] = { face.at<cv::Vec3b>(i - R, j - R)[2], face.at<cv::Vec3b>(i, j - R)[2], face.at<cv::Vec3b>(i + R, j - R)[2], face.at<cv::Vec3b>(i + R, j)[2], +face.at<cv::Vec3b>(i + R, j + R)[2], face.at<cv::Vec3b>(i, j + R)[2], face.at<cv::Vec3b>(i - R, j + R)[2], face.at<cv::Vec3b>(i - R, j)[2] };
			int value = 0;
			for (int k = 0; k < 8; k++)
			{
				if (face.at<cv::Vec3b>(i, j)[2] > neighbor[k])
				{
					value += pow(double(2), k);
				}
			}
			feature2.push_back(value);
		}
	}

	//Count the histogram
	for (int i = 0; i < 59; i++)
	{
		double x = count(feature2.begin(), feature2.end(), LBP[i]) / num;
		histogram.push_back(x);
	}
	return histogram;
}