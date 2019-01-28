#ifndef VideoREADER_H
#define VideoREADER_H


#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>

#include "Settings.h"
#include "frontend/Undistort.h"
#include "frontend/ImageRW.h"
#include "frontend/ImageAndExposure.h"

#include "internal/GlobalFuncs.h"
#include "internal/GlobalCalib.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace  ldso;
std::string  random_string( size_t length )
{
    auto randchar = []() -> char {const char charset[] ="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; const size_t max_index = (sizeof(charset) - 1); return charset[ rand() % max_index ];};
std::string str(length,0);
std::generate_n( str.begin(), length, randchar );
return  "_pfiles_tmp"+str;
}
class VideoFileReader
{
    cv::VideoCapture         vcap;
    cv::Mat _image,_imagegray;

 public:
    std::string getCalibFile(){return calibfile;}
    VideoFileReader(std::string path, std::string CvCalibFile  )
    {
        this->path = path;
         calibfile = ".dsocalib"+random_string(5);
        createDsoCalibFromCvCalib(CvCalibFile,calibfile);

        vcap.open(path);
        if (!vcap.isOpened())throw std::runtime_error("Could not open input video");
        undistort = Undistort::getUndistorterForFile(calibfile, "", "");
        widthOrg = undistort->getOriginalSize()[0];
        heightOrg = undistort->getOriginalSize()[1];
        width=undistort->getSize()[0];
        height=undistort->getSize()[1];


        // load timestamps if possible.
        loadTimestamps();
        printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());

    }
    ~VideoFileReader()
    {


        delete undistort;
    };


    void createDsoCalibFromCvCalib(std::string in,std::string out )
    {
         cv::FileStorage infile;
         std::cout<<"IN="<<in<<std::endl;
         infile.open(in,cv::FileStorage::READ);
         if (!infile.isOpened())throw std::runtime_error("Could not open "+in);
         cv::Mat cameraMatrix,distCoeff;
         cv::Mat cameraMatrix32,distCoeff32;
         float image_width,image_height;
         infile["camera_matrix"]>>cameraMatrix;
         infile["distortion_coefficients"]>>distCoeff;

         std::cout<<cameraMatrix<<std::endl;
         cameraMatrix.convertTo(cameraMatrix32, CV_32F);
         distCoeff.convertTo(distCoeff32, CV_32F);


         infile["image_width"]>>image_width;
         infile["image_height"]>>image_height;

         std::cout<<"OUT="<<out<<std::endl;
         std::ofstream ofile(out);
         if (!ofile)throw std::runtime_error("Could not open "+out);


         float fx=cameraMatrix32.at<float>(0,0);
         float cx=cameraMatrix32.at<float>(0,2);
         float fy=cameraMatrix32.at<float>(1,1);
         float cy=cameraMatrix32.at<float>(1,2);

         float k1=distCoeff32.ptr<float>(0)[0];
         float k2=distCoeff32.ptr<float>(0)[1];
         float p1=distCoeff32.ptr<float>(0)[2];
         float p2=distCoeff32.ptr<float>(0)[3];


         ofile<< "RadTan "<< fx/ double(image_width)<<" "<<fy/double(image_height)<<" "<< cx/ double(image_width) <<" "<<cy/ double(image_height)<<" "<<   k1<<" "<<k2<<" "<<p1<<" "<<p2<< std::endl;
         ofile<<image_width<<" "<<image_height<<std::endl;
         float scale= 1;//640./float(image_width);
         image_height*=scale;
         fx*=scale;
         cx*=scale;
         fy*=scale;
         cy*=scale;
         ofile<<  fx/ double(image_width)<<" "<<fy/double(image_height)<<" "<<cx/ double(image_width) <<" "<<cy/ double(image_height)<<" 0"<<std::endl;
         ofile<<image_width<<" "<<image_height<<std::endl;


    }


    Eigen::VectorXf getOriginalCalib()
    {
        return undistort->getOriginalParameter().cast<float>();
    }
    Eigen::Vector2i getOriginalDimensions()
    {
        return  undistort->getOriginalSize();
    }

    void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
    {
        K = undistort->getK().cast<float>();
        w = undistort->getSize()[0];
        h = undistort->getSize()[1];
    }

    void setGlobalCalibration()
    {
        int w_out, h_out;
        Eigen::Matrix3f K;
        getCalibMono(K, w_out, h_out);
        setGlobalCalib(w_out, h_out, K);
    }

    int getNumImages()
    {
        return  vcap.get(CV_CAP_PROP_FRAME_COUNT)-1;
    }

    double getTimestamp(int id)
    {
        if(timestamps.size()==0) return id*0.1f;
        if(id >= (int)timestamps.size()) return 0;
        if(id < 0) return 0;
        return timestamps[id];
    }


    void prepImage(int id, bool as8U=false)
    {

    }


    MinimalImageB* getImageRaw(int id)
    {
            return getImageRaw_internal(id,0);
    }

    ImageAndExposure* getImage(int id, bool forceLoadDirectly=false)
    {
        return getImage_internal(id, 0);
    }


    inline float* getPhotometricGamma()
    {
        if(undistort==0 || undistort->photometricUndist==0) return 0;
        return undistort->photometricUndist->getG();
    }


    // undistorter. [0] always exists, [1-2] only when MT is enabled.
    Undistort* undistort;
private:


    MinimalImageB* getImageRaw_internal(int id, int unused)
    {

        //if (slow) std::this_thread::sleep_for(std::chrono::milliseconds(500));
        vcap.set(CV_CAP_PROP_POS_FRAMES,id);
        bool res= vcap.grab();
        if (!res)return NULL;
        vcap.retrieve(_image);
        if (_image.channels()==3)
          cv::cvtColor(_image,_imagegray,CV_BGR2GRAY);
        else _imagegray=_image;

        MinimalImageB* img = new MinimalImageB(_imagegray.cols, _imagegray.rows);
        memcpy(img->data, _imagegray.data,_imagegray.rows*_imagegray.cols);
        return img;

       // return MinimalImageB(_image.cols,_image.rows,_imagegray.ptr<uchar>(0));


        if(!isZipped)
        {
            // CHANGE FOR ZIP FILE
            return IOWrap::readImageBW_8U(files[id]);
        }
        else
        {

            printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
            exit(1);

        }
    }


    ImageAndExposure* getImage_internal(int id, int unused)
    {
        MinimalImageB* minimg = getImageRaw_internal(id, 0);
        if (minimg==NULL)return NULL;
        ImageAndExposure* ret2 = undistort->undistort<unsigned char>(
                minimg,
                (exposures.size() == 0 ? 1.0f : exposures[id]),
                (timestamps.size() == 0 ? 0.0 : timestamps[id]));
        delete minimg;
        return ret2;
    }

    inline void loadTimestamps()
    {
        std::ifstream tr;
        std::string timesFile = path.substr(0,path.find_last_of('/')) + "/times.txt";
        tr.open(timesFile.c_str());
        while(!tr.eof() && tr.good())
        {
            std::string line;
            char buf[1000];
            tr.getline(buf, 1000);

            int id;
            double stamp;
            float exposure = 0;

            if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
            {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }

            else if(2 == sscanf(buf, "%d %lf", &id, &stamp))
            {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }
        }
        tr.close();

        // check if exposures are correct, (possibly skip)
        bool exposuresGood = ((int)exposures.size()==(int)getNumImages()) ;
        for(int i=0;i<(int)exposures.size();i++)
        {
            if(exposures[i] == 0)
            {
                // fix!
                float sum=0,num=0;
                if(i>0 && exposures[i-1] > 0) {sum += exposures[i-1]; num++;}
                if(i+1<(int)exposures.size() && exposures[i+1] > 0) {sum += exposures[i+1]; num++;}

                if(num>0)
                    exposures[i] = sum/num;
            }

            if(exposures[i] == 0) exposuresGood=false;
        }


        if((int)getNumImages() != (int)timestamps.size())
        {
            printf("set timestamps and exposures to zero!\n");
            exposures.clear();
            timestamps.clear();
        }

        if((int)getNumImages() != (int)exposures.size() || !exposuresGood)
        {
            printf("set EXPOSURES to zero!\n");
            exposures.clear();
        }

        printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(), (int)exposures.size());
    }




    std::vector<ImageAndExposure*> preloadedImages;
    std::vector<std::string> files;
    std::vector<double> timestamps;
    std::vector<float> exposures;

    int width, height;
    int widthOrg, heightOrg;

    std::string path;
    std::string calibfile;

    bool isZipped;


};
#endif
