#include <thread>
#include <clocale>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>

#include <glog/logging.h>
#include <iomanip>
#include "frontend/FullSystem.h"
#include "DatasetReader.h"
#include "video_reader.h"
using namespace std;
using namespace ldso;

/*********************************************************************************
 * This program demonstrates how to run LDSO in EUROC dataset
 * LDSO currently works on MAV_01-05 and V101, V102, V201
 * Note we don't have photometric calibration in EUROC so we should let the algorithm
 * estimate lighting parameter a,b for us
 *
 * You'd better set a start index greater than zero since DSO does not work well on blurred images
 *
 * Please specify the dataset directory below or by command line parameters
 *********************************************************************************/

std::string source = "/home/xiang/Dataset/EUROC/MH_01_easy/cam0";
std::string output_file = "./results.txt";
std::string calib = "./examples/EUROC/EUROC.txt";
std::string vocPath = "./vocab/orbvoc.dbow3";

int startIdx = 0;
int endIdx = 10000;

double rescale = 1;
bool reversePlay = false;
bool disableROS = false;
bool prefetch = false;
float playbackSpeed = 0;    // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload = false;
bool useSampleOutput = false;
bool firstRosSpin = false;

using namespace ldso;

void my_exit_handler(int s) {
    printf("Caught signal %d\n", s);
    exit(1);
}

void exitThread() {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    firstRosSpin = true;
    while (true) pause();
}


void settingsDefault(int preset) {
    printf("\n=============== PRESET Settings: ===============\n");
    if (preset == 0 || preset == 1) {
        printf("DEFAULT settings:\n"
               "- %s real-time enforcing\n"
               "- 2000 active points\n"
               "- 5-7 active frames\n"
               "- 1-6 LM iteration each KF\n"
               "- original image resolution\n", preset == 0 ? "no " : "1x");

        playbackSpeed = (preset == 0 ? 0 : 1);
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        setting_logStuff = false;
    }

    if (preset == 2 || preset == 3) {
        printf("FAST settings:\n"
               "- %s real-time enforcing\n"
               "- 800 active points\n"
               "- 4-6 active frames\n"
               "- 1-4 LM iteration each KF\n"
               "- 424 x 320 image resolution\n",
               preset == 0 ? "no " : "5x"); // preset cannot be zero, maybe wrong

        playbackSpeed = (preset == 2 ? 0 : 5);
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations = 4;
        setting_minOptIterations = 1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}


void parseArgument(char *arg) {
    int option;
    float foption;
    char buf[1000];


    if (1 == sscanf(arg, "sampleoutput=%d", &option)) {
        if (option == 1) {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "quiet=%d", &option)) {
        if (option == 1) {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "preset=%d", &option)) {
        settingsDefault(option);
        return;
    }


    if (1 == sscanf(arg, "rec=%d", &option)) {
        if (option == 0) {
            disableReconfigure = true;
            printf("DISABLE RECONFIGURE!\n");
        }
        return;
    }


    if (1 == sscanf(arg, "noros=%d", &option)) {
        if (option == 1) {
            disableROS = true;
            disableReconfigure = true;
            printf("DISABLE ROS (AND RECONFIGURE)!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "nolog=%d", &option)) {
        if (option == 1) {
            setting_logStuff = false;
            printf("DISABLE LOGGING!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "reversePlay=%d", &option)) {
        if (option == 1) {
            reversePlay = true;
            printf("REVERSE!\n");
        }
        return;
    }


    if (1 == sscanf(arg, "nomt=%d", &option)) {
        if (option == 1) {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }

    if (1 == sscanf(arg, "start=%d", &option)) {
        startIdx = option;
        printf("START AT %d!\n", startIdx);
        return;
    }
    if (1 == sscanf(arg, "end=%d", &option)) {
        endIdx = option;
        printf("END AT %d!\n", endIdx);
        return;
    }


    if (1 == sscanf(arg, "rescale=%f", &foption)) {
        rescale = foption;
        printf("RESCALE %f!\n", rescale);
        return;
    }

    if (1 == sscanf(arg, "speed=%f", &foption)) {
        playbackSpeed = foption;
        printf("PLAYBACK SPEED %f!\n", playbackSpeed);
        return;
    }

    if (1 == sscanf(arg, "output=%s", &buf)) {
        output_file = buf;
        LOG(INFO) << "output set to " << output_file << endl;
        return;
    }

    if (1 == sscanf(arg, "calib=%s", buf)) {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }



    if (1 == sscanf(arg, "mode=%d", &option)) {
        if (option != 1) {
            LOG(ERROR) << "EuRoC does not have photometric intrinsics! I will exit!" << endl;
            exit(-1);
        }
        return;
    }

    if (1 == sscanf(arg, "loopclosing=%d", &option)) {
        if (option == 1) {
            setting_enableLoopClosing = true;
        } else {
            setting_enableLoopClosing = false;
        }
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
}
class CmdLineParser{int argc; char **argv;public:
CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}
bool operator[] ( std::string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( std::string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    }
std::string operator()(std::string param,std::string defvalue=""){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( std::string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }
};

void  printResult(const string &filename, const std::vector< std::pair<int,SE3> > &poses) {


    std::ofstream myfile(filename);
    myfile << std::setprecision(15);

    for (auto &fr : poses) {
        SE3 Twc;
        Sim3 Swc;

            Twc = fr.second.inverse();

        myfile << fr.first<<
               " " << Twc.translation().transpose() <<
               " " << Twc.so3().unit_quaternion().x() <<
               " " << Twc.so3().unit_quaternion().y() <<
               " " << Twc.so3().unit_quaternion().z() <<
               " " << Twc.so3().unit_quaternion().w() << "\n";
    }
    myfile.close();
}
int main(int argc, char **argv) {

    CmdLineParser cml(argc,argv);
    if(argc<5 || cml["-h"]){
        printf("Usage :  videofile  cameraparams.yml path_to_vocabulary outposes [-noX] [-realTime] [-end X]\n");
        return -1;
    }
    settingsDefault(0);
    source=argv[1];
    calib=argv[2];
    vocPath=argv[3];
    output_file=argv[4];
    FLAGS_colorlogtostderr = true;
    setting_maxAffineWeight = 0.1;  // don't use affine brightness weight in Euroc!
    disableAllDisplay = cml["-noX"];
    int end=std::numeric_limits<int>::max();
    if(cml["-end"]) end=stoi(cml("-end"));

//    for (int i = 1; i < argc; i++)
//        parseArgument(argv[i]);

    // EuRoC has no photometric calibration
    printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
    setting_photometricCalibration = 0;
    setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
    setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).

    // hook crtl+C.
    thread exThread = thread(exitThread);

    VideoFileReader* reader = new VideoFileReader(source,calib );
    reader->setGlobalCalibration();


//    shared_ptr<ImageFolderReader> reader(
//            new ImageFolderReader(ImageFolderReader::EUROC, source, calib, "", ""));    // no gamma and vignette

//    reader->setGlobalCalibration();

    int lstart = startIdx;
    int lend = endIdx;
    int linc = 1;



    shared_ptr<ORBVocabulary> voc(new ORBVocabulary());
    voc->load(vocPath);

    shared_ptr<FullSystem> fullSystem(new FullSystem(voc));
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    //rms
//     if( !cml["-realTime"])    playbackSpeed=0;
//     else playbackSpeed=20;
    fullSystem->linearizeOperation = (playbackSpeed == 0);

    shared_ptr<PangolinDSOViewer> viewer = nullptr;
    if (!disableAllDisplay) {
        viewer = shared_ptr<PangolinDSOViewer>(new PangolinDSOViewer(wG[0], hG[0], false));
        fullSystem->setViewer(viewer);
    } else {
        LOG(INFO) << "visualization is diabled!" << endl;
    }
    cout<<"WRITING OUTPUT TO"<<output_file<<endl;

    if(cml["-timefile"]){
        std::string commd="date > "+cml("-timefile");
        system(commd.c_str());
    }
    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {



    ImageAndExposure *img ;
    int ii=0;
        while((img  = reader->getImage(ii))!=NULL){
            img->timestamp=ii;

           fullSystem->addActiveFrame(img, ii++);
            delete img;

            if (fullSystem->initFailed || setting_fullResetRequested) {
                if (ii < 250 || setting_fullResetRequested) {
                    LOG(INFO) << "RESETTING!";
                    fullSystem = shared_ptr<FullSystem>(new FullSystem(voc));
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed == 0);
                    if (viewer) {
                        viewer->reset();
                        sleep(1);
                        fullSystem->setViewer(viewer);
                    }
                    setting_fullResetRequested = false;
                }
            }

            if (fullSystem->isLost) {
                LOG(INFO) << "Lost!";
                break;
            }
            if( ii>=end)break;
        }
        if(cml["-timefile"]){
            std::string commd="date >> "+cml("-timefile");
            system(commd.c_str());
        }
        //now, go again only for tracking
        ii=0;
        std::vector< std::pair<int,SE3> > poses;
               while((img  = reader->getImage(ii))!=NULL){
                   img->timestamp=ii;
                   cout<<"TIME="<<ii<<endl;

                  SE3 pose=fullSystem->addActiveFrame(img, ii++);
                  poses.push_back({ii-1,pose});
                  delete img;

                   if (fullSystem->initFailed || setting_fullResetRequested) {
                       if (ii < 250 || setting_fullResetRequested) {
                           LOG(INFO) << "RESETTING!";
                           fullSystem = shared_ptr<FullSystem>(new FullSystem(voc));
                           fullSystem->setGammaFunction(reader->getPhotometricGamma());
                           fullSystem->linearizeOperation = (playbackSpeed == 0);
                           if (viewer) {
                               viewer->reset();
                               sleep(1);
                               fullSystem->setViewer(viewer);
                           }
                           setting_fullResetRequested = false;
                       }
                   }

                   if (fullSystem->isLost) {
                       LOG(INFO) << "Lost!";
                       break;
                   }
                if( ii>=end)
                    break;
               }
               //
               printResult(output_file,poses);
               if(cml["-timefile"]){
                   std::string commd="date >> "+cml("-timefile");
                   system(commd.c_str());
               }
               fullSystem->blockUntilMappingIsFinished();
               sleep(10);
    });

    if (viewer)
        viewer->run();  // mac os should keep this in main thread.

    runthread.join();

//    fullSystem->printResult(output_file, true);
//    fullSystem->printResult(output_file+".noloop", false);

    LOG(INFO) << "EXIT NOW!";
    return 0;
}
