#include <iostream>
#include <stdio.h>
#include <string.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <pangolin/pangolin.h>
#include <pangolin/utils/timer.h>

#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "depth_sources/image_depth_source.h"
#include "geometry/plane_fitting.h"
#include "img_proc/img_ops.h"
#include "optimization/priors.h"
#include "tracker.h"
#include "util/dart_io.h"
#include "util/gl_dart.h"
#include "util/image_io.h"
#include "util/ostream_operators.h"
#include "util/string_format.h"
#include "visualization/color_ramps.h"
#include "visualization/data_association_viz.h"
#include "visualization/gradient_viz.h"
#include "visualization/sdf_viz.h"
//include the urdf support
#include "model/read_model_urdf.h"
// #include <priors.hpp>

#define EIGEN_DONT_ALIGN

#define DEPTH_SOURCE_LCM
#define DEPTH_SOURCE_LCM_MULTISENSE
//#define DEPTH_SOURCE_LCM_XTION
// read depth (not disparity) images from MultiSense SL, use specific camera parameters
//#define DEPTH_SOURCE_IMAGE_MULTISENSE

//#include <dart_lcm/lcm_state_publish.hpp>
//#include <dart_lcm/lcm_frame_pose_publish.hpp>

#ifdef DEPTH_SOURCE_LCM
    //#include <dart_lcm/dart_lcm_depth_provider.hpp>
#endif

#define ENABLE_URDF
#define ENABLE_LCM_JOINTS

// what are these????????????????????????????????????????????????????
#define LCM_CHANNEL_ROBOT_STATE "EST_ROBOT_STATE"
//#define LCM_CHANNEL_ROBOT_STATE "EST_ROBOT_STATE_ORG"
#define LCM_CHANNEL_DART_PREFIX "DART_"

using namespace std;

enum PointColorings {
    PointColoringNone = 0,
    PointColoringRGB,
    PointColoringErr,
    PointColoringDA,
    NumPointColorings
};

enum DebugImgs {
    DebugColor=0,
    DebugObsDepth,
    DebugPredictedDepth,
    DebugObsToModDA,
    DebugModToObsDA,
    DebugObsToModErr,
    DebugModToObsErr,
    DebugJTJ,
    DebugN
};

//not changed param
const static int panelWidth = 180;

//not changed func
void setSlidersFromTransform(dart::SE3& transform, pangolin::Var<float>** sliders) {
    *sliders[0] = transform.r0.w; transform.r0.w = 0;
    *sliders[1] = transform.r1.w; transform.r1.w = 0;
    *sliders[2] = transform.r2.w; transform.r2.w = 0;
    dart::se3 t = dart::se3FromSE3(transform);
    *sliders[3] = t.p[3];
    *sliders[4] = t.p[4];
    *sliders[5] = t.p[5];
}

//not changed func
void setSlidersFromTransform(const dart::SE3& transform, pangolin::Var<float>** sliders) {
    dart::SE3 mutableTransform = transform;
    setSlidersFromTransform(mutableTransform,sliders);
}

//get a pose without pose reduction
dart::Pose noneReductionPose(const dart::HostOnlyModel &model) {
    std::vector<float> minJoints;
    std::vector<float> maxJoints;
    std::vector<std::string> jointNames;
    for (int i = 0; i < model.getNumJoints(); i++) {
        minJoints.push_back(model.getJointMin(i));
        maxJoints.push_back(model.getJointMax(i));
        jointNames.push_back(model.getJointName(i));
    }
    return dart::Pose(new dart::NullReduction(model.getNumJoints(),
                    minJoints.data(), maxJoints.data(), jointNames.data()));
}

// can be commented out
// print all joint positions stored in a pose
void printPoseJoints(dart::Pose &pose) {
    const int ndims = pose.getReducedArticulatedDimensions();
    std::cout<<"reduced joints -- " << ndims << std::endl;
    for(int i = 0; i < ndims; i++) {
        std::cout << i << " " <<pose.getReducedName(i)<<": "<<pose.getReducedArticulation()[i]<<std::endl;
    }
}

// not changed param
static float3 initialTableNorm = make_float3(0.0182391, 0.665761, -0.745942);
static float initialTableIntercept = -0.705196;

int main(int argc, char *argv[]) {
    const std::string videoLoc = "../video/"; // need to be deleted 
    // -=-=-=- initializations -=-=-=- //not changed from the example
    cudaSetDevice(0);
    cudaDeviceReset();

    pangolin::CreateWindowAndBind("Main",640+4*panelWidth+1,2*480+1);

    glewInit();
    glutInit(&argc, argv);
    dart::Tracker tracker;

    // -=-=-=- pangolin window setup -=-=-=-

    pangolin::CreatePanel("lim").SetBounds(0.0,1.0,1.0,pangolin::Attach::Pix(-panelWidth));
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panelWidth));
    pangolin::CreatePanel("opt").SetBounds(0.0,1.0,pangolin::Attach::Pix(panelWidth), pangolin::Attach::Pix(2*panelWidth));
    pangolin::CreatePanel("pose").SetBounds(0.0,1.0,pangolin::Attach::Pix(-panelWidth), pangolin::Attach::Pix(-2*panelWidth));

    int glWidth = 640;
    int glHeight= 480;
    int glFL = 400;
    int glPPx = glWidth/2;
    int glPPy = glHeight/2;

    pangolin::OpenGlMatrixSpec glK = pangolin::ProjectionMatrixRUB_BottomLeft(glWidth,glHeight,glFL,glFL,glPPx,glPPy,0.01,1000);
    pangolin::OpenGlMatrix viewpoint = pangolin::ModelViewLookAt(0, 0, 0.05, 0, -0.1, 0.2, pangolin::AxisY);
    //pangolin::OpenGlMatrix viewpoint = pangolin::ModelViewLookAt(0, 0, 0.05, 0, 0, 0.2, pangolin::AxisY);
    // the MultiSense and the Xtion have different oriented frames
    
    //for DEPTH_SOURCE_LCM_MULTISENSE
    // Z forward, Y up
    pangolin::OpenGlRenderState camState(glK, viewpoint);


    // for DEPTH_SOURCE_LCM_XTION
    // Z forward, Y down
    // pangolin::OpenGlRenderState camState(pangolin::OpenGlMatrix::RotateZ(M_PI)*glK, viewpoint);
    
    //not changed param
    pangolin::View & camDisp = pangolin::Display("cam").SetAspect(640.0f/480.0f).SetHandler(new pangolin::Handler3D(camState));
    pangolin::View & imgDisp = pangolin::Display("img").SetAspect(640.0f/480.0f);
    pangolin::GlTexture imgTexDepthSize(320,240);
    pangolin::GlTexture imgTexPredictionSize(160,120);

    //not changed param
    pangolin::DataLog infoLog;
    {
        std::vector<std::string> infoLogLabels;
        infoLogLabels.push_back("errObsToMod");
        infoLogLabels.push_back("errModToObs");
        infoLogLabels.push_back("stabilityThreshold");
        infoLogLabels.push_back("resetThreshold");
        infoLog.SetLabels(infoLogLabels);
    }

    //not changed param
    pangolin::Display("multi")
            .SetBounds(1.0, 0.0, pangolin::Attach::Pix(2*panelWidth), pangolin::Attach::Pix(-2*panelWidth))
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(camDisp)
            //.AddDisplay(infoPlotter)
            .AddDisplay(imgDisp);
            
    //debug use
    imgDisp.Show(false);

    // not change param
    float defaultModelSdfPadding = 0.07;

    std::vector<pangolin::Var<float> *> sizeVars;

    // not defined in the original code 
    //for DEPTH_SOURCE_IMAGE_TEST
    /*
    dart::ImageDepthSource<uint16_t,uchar3> * depthSource = new dart::ImageDepthSource<uint16_t,uchar3>();
    depthSource->initialize("../test_video/",dart::IMAGE_PNG, make_float2(525/2,525/2),make_float2(320,240), 640,480,0.001);
    //depthSource->initialize("../test_video2/",dart::IMAGE_PNG, make_float2(525/2,525/2),make_float2(2,2), 5,4,0.001);
    */

    // commented out in the original code 
    //for DEPTH_SOURCE_IMAGE_MULTISENSE
    /*
    dart::ImageDepthSource<uint16_t,uchar3> * depthSource = new dart::ImageDepthSource<uint16_t,uchar3>();
    depthSource->initialize("../depth_grasping_bottle",dart::IMAGE_PNG, make_float2(556.183166504, 556.183166504),make_float2(512,512), 1024, 1024, 0.001);
    */


    // Depth source from the CAMERA
    //for DEPTH_SOURCE_LCM_MULTISENSE_HA // uncomment here!!!!!!!
    // Valkyrie Unit D, MultiSense SL
    /*
    dart::StereoCameraParameter val_multisense;                 // is this from lcm??????????? not in original file
    val_multisense.focal_length = make_float2(556.183166504, 556.183166504);
    val_multisense.camera_center = make_float2(512, 512);
    val_multisense.baseline = 0.07;
    val_multisense.width = 1024;
    val_multisense.height = 1024;
    val_multisense.subpixel_resolution = 1.0/16.0;

    // initialise LCM depth source and listen on channel "CAMERA" in a separate thread
    dart::LCM_DepthSource<float,uchar3> *depthSource = new dart::LCM_DepthSource<float,uchar3>(val_multisense); // not in the original lib
    depthSource->subscribe_images("CAMERA");
    //depthSource->subscribe_images("CAMERA_FILTERED");
    depthSource->setMaxDepthDistance(1.0); // meter

    const std::string cam_frame_name = "left_camera_optical_frame_joint";
    */

    dart::ImageDepthSource<ushort,uchar3> * depthSource = new dart::ImageDepthSource<ushort,uchar3>();
    depthSource->initialize(videoLoc+"/depth",dart::IMAGE_PNG,
                            make_float2(525/2,525/2),make_float2(160,120),
                            320,240,0.001,0);
    //not changed param ->
    tracker.addDepthSource(depthSource);
    dart::Optimizer & optimizer = *tracker.getOptimizer();

    const static int obsSdfSize = 64;
    const static float obsSdfResolution = 0.01*32/obsSdfSize;
    const static float defaultModelSdfResolution = 2e-3; //1.5e-3;
    const static float3 obsSdfOffset = make_float3(0,0,0.1);

    pangolin::Var<float> modelSdfResolution("lim.modelSdfResolution",defaultModelSdfResolution,defaultModelSdfResolution/2,defaultModelSdfResolution*2);
    pangolin::Var<float> modelSdfPadding("lim.modelSdfPadding",defaultModelSdfPadding,defaultModelSdfPadding/2,defaultModelSdfPadding*2);

    // <- not changed param

    // original model
    //const std::string urdf_model_path = "../models/baxter_description/urdf/baxter.urdf";
    const std::string urdf_model_path = "../models/test/my_robot.urdf";
    // add Baxter
    dart::HostOnlyModel val = dart::readModelURDF(urdf_model_path, "link1");                // base for the baxter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    std::cout << "found robot: " << val.getName() << std::endl;

    // initialize pose with 0 joint values and no full to reduced mapping
    dart::Pose val_pose = noneReductionPose(val);
    val_pose.zero();

    // joints/frame IDs for finding transformations
    // Uncomment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // const int val_cam_frame_id = val.getJointFrame(val.getJointIdByName(cam_frame_name));

    // track bottle
    dart::HostOnlyModel bottle = dart::readModelURDF("../models/test/my_robot.urdf");
    tracker.addModel(bottle, 0.5*modelSdfResolution, modelSdfPadding, 64);
    // initial bottle pose in camera coordinate system, transformation camera to bottle
    // rotation according to Tait-Bryan angles: Z_1 Y_2 X_3
    // e.g. first: rotation around Z-axis, second: rotation around Y-axis, third: rotation around X-axis

    // multisense
    // lcmlog__2016-04-20__18-51-14-226028__yy-wxm-table-grasping-left-hand
    // t>1150s
    const dart::SE3 T_cb = dart::SE3FromTranslation(0.244, -0.3036, 0.5952) * dart::SE3FromEuler(make_float3(0.2244, -0.594, -0.6732));
    // const dart::SE3 T_cb = dart::SE3FromTranslation(0.1131, -0.07738, 0.6071) * dart::SE3FromRotationX(0.4862) * dart::SE3FromRotationY(-2.02+1.04) * dart::SE3FromRotationZ(2.02);

    // track subparts of Valkyrie
    const std::vector<uint8_t> colour_estimated_model = {255, 200, 0}; // yellow-orange
    dart::HostOnlyModel val_torso = dart::readModelURDF(urdf_model_path, "link1", "obj", colour_estimated_model); // what is torso???? obj???? check the method

    // const int val_torso_cam_frame_id = val_torso.getJointFrame(val_torso.getJointIdByName(cam_frame_name));

    tracker.addModel(val_torso,             // dart::hostOnlyModel
                     modelSdfResolution,    // modelSdfResolution, def = 0.002
                     modelSdfPadding,       // modelSdfPadding, def = 0.07
                     obsSdfSize,
                     obsSdfResolution,
                     make_float3(-0.5*obsSdfSize*obsSdfResolution) + obsSdfOffset,
                     0,         // poseReduction
                     1e5,       // collisionCloudDensity (def = 1e5)
                     true      // cacheSdfs
                     );

    // prevent movement of the camera frame by enforcing no transformation
    //dart::NoCameraMovementPrior val_cam(tracker.getModelIDbyName("valkyrie")); // method from lcm
    // TODO -- DO I HAVE THIS ?
    //tracker.addPrior(&val_cam);
    std::cout<<"added models: "<<tracker.getNumModels()<<std::endl;
    
    // not changed param -->
    std::vector<pangolin::Var<float> * *> poseVars;

    pangolin::Var<bool> sliderControlled("pose.sliderControl",false,true);
    for (int m=0; m<tracker.getNumModels(); ++m) {

        const int dimensions = tracker.getModel(m).getPoseDimensionality();

        pangolin::Var<float> * * vars = new pangolin::Var<float> *[dimensions];
        poseVars.push_back(vars);
        poseVars[m][0] = new pangolin::Var<float>(dart::stringFormat("pose.%d x",m),0,-0.5,0.5);
        poseVars[m][1] = new pangolin::Var<float>(dart::stringFormat("pose.%d y",m),0,-0.5,0.5);
        poseVars[m][2] = new pangolin::Var<float>(dart::stringFormat("pose.%d z",m),0.3,0.5,1.5);
        poseVars[m][3] = new pangolin::Var<float>(dart::stringFormat("pose.%d wx",m),    0,-M_PI,M_PI);
        poseVars[m][4] = new pangolin::Var<float>(dart::stringFormat("pose.%d wy",m),    0,-M_PI,M_PI);
        poseVars[m][5] = new pangolin::Var<float>(dart::stringFormat("pose.%d wz",m), M_PI,-M_PI,M_PI);

        const dart::Pose & pose = tracker.getPose(m);
        for (int i=0; i<pose.getReducedArticulatedDimensions(); ++i) {
            poseVars[m][i+6] = new pangolin::Var<float>(dart::stringFormat("pose.%d %s",m,pose.getReducedName(i).c_str()),0,pose.getReducedMin(i),pose.getReducedMax(i));
        }

    }
    // pangolin variables
    static pangolin::Var<bool> trackFromVideo("ui.track",true,false,true);
    static pangolin::Var<bool> stepVideo("ui.stepVideo",false,false);
    static pangolin::Var<bool> stepVideoBack("ui.stepVideoBack",false,false);

    // add for urdf support
    static pangolin::Var<bool> resetRobotPose("ui.resetRobotPose",false,false);
    static pangolin::Var<bool> useReportedPose("ui.useReportedPose",false,true);

    static pangolin::Var<float> sigmaPixels("ui.sigmaPixels",3.0,0.01,4);
    static pangolin::Var<float> sigmaDepth("ui.sigmaDepth",0.1,0.001,1);
    static pangolin::Var<float> focalLength("ui.focalLength",depthSource->getFocalLength().x,0.8*depthSource->getFocalLength().x,1.2*depthSource->getFocalLength().x);//475,525); //525.0,450.0,600.0);     |       //static pangolin::Var<float> focalLength_y("ui.focalLength_y",depthSource->getFocalLength().y, 500, 1500);
    static pangolin::Var<bool> showCameraPose("ui.showCameraPose",false,true);
    static pangolin::Var<bool> showEstimatedPose("ui.showEstimate",true,true);//static pangolin::Var<bool> showEstimatedPose("ui.showEstimate",false,true);
    static pangolin::Var<bool> showReported("ui.showReported",true,true); //static pangolin::Var<bool> showReported("ui.showReported",false,true);

    // for JUSTIN
    // static pangolin::Var<bool> showTablePlane("ui.showTablePlane",false,true);

    static pangolin::Var<bool> showVoxelized("ui.showVoxelized",false,true);
    static pangolin::Var<float> levelSet("ui.levelSet",0.0,-10.0,10.0);
    static pangolin::Var<bool> showTrackedPoints("ui.showPoints",true,true);
    // for DEPTH_SOURCE_LCM
    static pangolin::Var<bool> showPointColour("ui.showColour",true,true);

    static pangolin::Var<int> pointColoringObs("ui.pointColoringObs",0,0,NumPointColorings-1);
    static pangolin::Var<int> pointColoringPred("ui.pointColoringPred",0,0,NumPointColorings-1);
    // for JUSTIN
    // static pangolin::Var<float> planeOffset("ui.planeOffset",-0.03,-0.05,0);

    static pangolin::Var<int> debugImg("ui.debugImg",DebugN,0,DebugN);

    static pangolin::Var<bool> showObsSdf("ui.showObsSdf",false,true);
    static pangolin::Var<bool> showPredictedPoints("ui.showPredictedPoints",false,true);
    static pangolin::Var<bool> showCollisionClouds("ui.showCollisionClouds",false,true);
    // new, record button
    static pangolin::Var<bool> record("ui.Record Start/Stop",false,false);
    static pangolin::Var<float> fps("ui.fps",0);

    // optimization options
    pangolin::Var<bool> iterateButton("opt.iterate",false,false);
    pangolin::Var<int> itersPerFrame("opt.itersPerFrame",3,0,30);
    pangolin::Var<float> normalThreshold("opt.normalThreshold",-1.01,-1.01,1.0);
    pangolin::Var<float> distanceThreshold("opt.distanceThreshold",0.035,0.0,0.1);
    pangolin::Var<float> handRegularization("opt.handRegularization",0.1,0,10); // 1.0
    pangolin::Var<float> objectRegularization("opt.objectRegularization",0.1,0,10); // 1.0
    pangolin::Var<float> resetInfoThreshold("opt.resetInfoThreshold",1.0e-5,1e-5,2e-5);
    pangolin::Var<float> stabilityThreshold("opt.stabilityThreshold",7.5e-6,5e-6,1e-5);
    pangolin::Var<float> lambdaModToObs("opt.lambdaModToObs",0.5,0,1);
    pangolin::Var<float> lambdaObsToMod("opt.lambdaObsToMod",1,0,1);

    pangolin::Var<float> infoAccumulationRate("opt.infoAccumulationRate",0.1,0.0,1.0); // 0.8
    pangolin::Var<float> maxRotationDamping("opt.maxRotationalDamping",50,0,200);
    pangolin::Var<float> maxTranslationDamping("opt.maxTranslationDamping",5,0,10);

    /* for JUSTIN    
    pangolin::Var<float> tableNormX("opt.tableNormX",initialTableNorm.x,-1,1);
    pangolin::Var<float> tableNormY("opt.tableNormY",initialTableNorm.y,-1,1);
    pangolin::Var<float> tableNormZ("opt.tableNormZ",initialTableNorm.z,-1,1);
    pangolin::Var<float> tableIntercept("opt.tableIntercept",initialTableIntercept,-1,1);
    static pangolin::Var<bool> fitTable("opt.fitTable",true,true);
    static pangolin::Var<bool> subtractTable("opt.subtractTable",true,true);
    */

    /* USE_CONTACT_PRIOR
    static pangolin::Var<bool> * contactVars[10];
    contactVars[0] = new pangolin::Var<bool>("lim.contactThumbR",false,true);
    contactVars[1] = new pangolin::Var<bool>("lim.contactIndexR",false,true);
    contactVars[2] = new pangolin::Var<bool>("lim.contactMiddleR",false,true);
    contactVars[3] = new pangolin::Var<bool>("lim.contactRingR",false,true);
    contactVars[4] = new pangolin::Var<bool>("lim.contactLittleR",false,true);
    contactVars[5] = new pangolin::Var<bool>("lim.contactThumbL",false,true);
    contactVars[6] = new pangolin::Var<bool>("lim.contactIndexL",false,true);
    contactVars[7] = new pangolin::Var<bool>("lim.contactMiddleL",false,true);
    contactVars[8] = new pangolin::Var<bool>("lim.contactRingL",false,true);
    contactVars[9] = new pangolin::Var<bool>("lim.contactLittleL",false,true);
    bool anyContact = false;
    */

//    // not changed -------------------->
//     int fpsWindow = 10;
//     pangolin::basetime lastTime = pangolin::TimeNow();

//     const int depthWidth = depthSource->getDepthWidth();
//     const int depthHeight = depthSource->getDepthHeight();

//     const int predWidth = tracker.getPredictionWidth();
//     const int predHeight = tracker.getPredictionHeight();

//     dart::MirroredVector<uchar3> imgDepthSize(depthWidth*depthHeight);
//     dart::MirroredVector<uchar3> imgPredSize(predWidth*predHeight);
//     dart::MirroredVector<const uchar3 *> allSdfColors(tracker.getNumModels());
//     for (int m = 0; m < tracker.getNumModels(); m++) {
//         allSdfColors.hostPtr()[m] = tracker.getModel(m).getDeviceSdfColors();
//     }
//     allSdfColors.syncHostToDevice();

//     // set up VBO to display point cloud
//     GLuint pointCloudVbo,pointCloudColorVbo,pointCloudNormVbo;
//     glGenBuffersARB(1,&pointCloudVbo);
//     glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudVbo);
//     glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(float4),tracker.getHostVertMap(),GL_DYNAMIC_DRAW_ARB);
//     glGenBuffersARB(1,&pointCloudColorVbo);
//     glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
//     glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(uchar3),imgDepthSize.hostPtr(),GL_DYNAMIC_DRAW_ARB);
//     glGenBuffersARB(1,&pointCloudNormVbo);
//     glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudNormVbo);
//     glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(float4),tracker.getHostNormMap(),GL_DYNAMIC_DRAW_ARB);


//     dart::OptimizationOptions & opts = tracker.getOptions();
//     opts.lambdaObsToMod = 1;
//     memset(opts.lambdaIntersection.data(),0,tracker.getNumModels()*tracker.getNumModels()*sizeof(float));
//     opts.regularization[0] = opts.regularization[1] = opts.regularization[2] = 0.01;
//     // <-------------------- not changed 

//     // get references to model and its pose for tracking
//     dart::MirroredModel & val_torso_mm = tracker.getModel(tracker.getModelIDbyName("valkyrie")); // how the mirror model works ????????????????????????????
//     dart::Pose & val_torso_pose = tracker.getPose("valkyrie");
//     dart::Pose & bottle_pose = tracker.getPose("bottle");

//     // measures joint values for reported robot configuration
//     dart::LCM_JointsProvider lcm_joints;
//     lcm_joints.setJointNames(val);
//     // listen on channel "EST_ROBOT_STATE" in a separate thread
//     lcm_joints.subscribe_robot_state(LCM_CHANNEL_ROBOT_STATE);

//     dart::LCM_StatePublish lcm_robot_state(LCM_CHANNEL_ROBOT_STATE, LCM_CHANNEL_DART_PREFIX, val_torso_pose); // need lcm code !!!!!!!!!!!
//     dart::LCM_FramePosePublish lcm_frame_pub("DART", val, val_torso_mm); // need lcm code !!!!!!!!!!!

//     dart::MirroredModel & bottle_mm = tracker.getModel(tracker.getModelIDbyName("bottle"));
//     dart::LCM_FramePosePublish lcm_object_frame_pub("DART", bottle, bottle_mm); // need lcm code !!!!!!!!!!!!



//     // -=-=-=-=- set up initial poses -=-=-=-=-

//     // wait to get initial configuration of robot from LCM thread
//     usleep(100000);
//     // set initial state of tracked model
//     val_torso_pose.setReducedArticulation(lcm_joints.getJointsNameValue());
//     val_torso_mm.setPose(val_torso_pose);
//     dart::SE3 Tmc = val_torso_mm.getTransformModelToFrame(val_torso_cam_frame_id);
//     val_torso_pose.setTransformModelToCamera(Tmc);
//     bottle_pose.setTransformModelToCamera(T_cb);
//     bottle.setPose(bottle_pose);

//     // ------------------- main loop ---------------------
//     for (int pangolinFrame=1; !pangolin::ShouldQuit(); ++pangolinFrame) {

//         if (pangolin::HasResized()) {
//             pangolin::DisplayBase().ActivateScissorAndClear();
//         }

//         tracker.stepForward();

//         // get reported Valkyrie configuration
//         val_pose.setReducedArticulation(lcm_joints.getJointsNameValue());
//         // transform coordinate origin to camera image centre
//         dart::SE3 Tmc = val.getTransformModelToFrame(val_cam_frame_id);
//         val_pose.setTransformModelToCamera(Tmc);

//         if(pangolin::Pushed(resetRobotPose) || useReportedPose) {
//             val_torso_pose.setReducedArticulation(lcm_joints.getJointsNameValue());

//             val_torso_mm.setPose(val_torso_pose);
//             dart::SE3 Tmc = val_torso_mm.getTransformModelToFrame(val_torso_cam_frame_id);
//             val_torso_pose.setTransformModelToCamera(Tmc);
//         }

//         opts.focalLength = focalLength;
//         opts.normThreshold = normalThreshold;
//         for (int m=0; m<tracker.getNumModels(); ++m) {
//             opts.distThreshold[m] = distanceThreshold;
//         }
//         opts.regularization[0] = opts.regularization[1] = opts.regularization[2] = 0.01;
//         opts.lambdaObsToMod = lambdaObsToMod;
//         opts.lambdaModToObs = lambdaModToObs;
//         opts.debugObsToModDA = pointColoringObs == PointColoringDA || (debugImg == DebugObsToModDA);
//         opts.debugModToObsDA = debugImg == DebugModToObsDA;
//         opts.debugObsToModErr = ((pointColoringObs == PointColoringErr) || (debugImg == DebugObsToModErr));
//         opts.debugModToObsErr = ((pointColoringPred == PointColoringErr) || (debugImg == DebugModToObsErr));
//         opts.debugJTJ = (debugImg == DebugJTJ);
//         opts.numIterations = itersPerFrame;

//         if (pangolin::Pushed(stepVideoBack)) {
//             tracker.stepBackward();
//         }

//         bool iteratePushed = Pushed(iterateButton);

//         if (pangolinFrame % fpsWindow == 0) {
//             pangolin::basetime time = pangolin::TimeNow();
//             if (trackFromVideo) {
//                 static int totalFrames = 0;
//                 static double totalTime = 0;
//                 totalFrames += fpsWindow;
//                 totalTime += pangolin::TimeDiff_s(lastTime,time);
//                 fps = totalFrames / totalTime;
//             } else {
//                 fps = fpsWindow / pangolin::TimeDiff_s(lastTime,time);
//             }
//             lastTime = time;
//         }

//         //////////////////////////////////////////////////////////////////////////////////////////////////////////
//         //                                                                                                      //
//         //                                          Process this frame                                          //
//         //                                                                                                      //
//         //////////////////////////////////////////////////////////////////////////////////////////////////////////
//         // not changed param -------------------->
//         {
//             static pangolin::Var<bool> filteredNorms("ui.filteredNorms",false,true);
//             static pangolin::Var<bool> filteredVerts("ui.filteredVerts",false,true);

//             if (filteredNorms.GuiChanged()) {
//                 tracker.setFilteredNorms(filteredNorms);
//             } else if (filteredVerts.GuiChanged()) {
//                 tracker.setFilteredVerts(filteredVerts);
//             } else if (sigmaDepth.GuiChanged()) {
//                 tracker.setSigmaDepth(sigmaDepth);
//             } else if (sigmaPixels.GuiChanged()) {
//                 tracker.setSigmaPixels(sigmaPixels);
//             }

//             // update pose based on sliders
//             if (sliderControlled) {
//                 for (int m=0; m<tracker.getNumModels(); ++m) {
//                     for (int i=0; i<tracker.getPose(m).getReducedArticulatedDimensions(); ++i) {
//                         tracker.getPose(m).getReducedArticulation()[i] = *poseVars[m][i+6];
//                     }
//                     tracker.getPose(m).setTransformModelToCamera(dart::SE3Fromse3(dart::se3(*poseVars[m][0],*poseVars[m][1],*poseVars[m][2],0,0,0))*
//                             dart::SE3Fromse3(dart::se3(0,0,0,*poseVars[m][3],*poseVars[m][4],*poseVars[m][5])));
//                     tracker.updatePose(m);
//                 }
//             }

//             // run optimization method
//             if (trackFromVideo || iteratePushed ) {

//                 // workaround: we need to wait 1 frame before starting optimization
//                 // otherwise, the no movement prior produces a wrong update
//                 if(pangolinFrame>1)
//                     tracker.optimizePoses();

//                 // update accumulated info
//                 for (int m=0; m<tracker.getNumModels(); ++m) {
//                     const Eigen::MatrixXf & JTJ = *tracker.getOptimizer()->getJTJ(m);
//                     if (JTJ.rows() == 0) { continue; }
//                     Eigen::MatrixXf & dampingMatrix = tracker.getDampingMatrix(m);
//                     for (int i=0; i<3; ++i) {
//                         dampingMatrix(i,i) = std::min((float)maxTranslationDamping,dampingMatrix(i,i) + infoAccumulationRate*JTJ(i,i));
//                     }
//                     for (int i=3; i<tracker.getPose(m).getReducedDimensions(); ++i) {
//                         dampingMatrix(i,i) = std::min((float)maxRotationDamping,dampingMatrix(i,i) + infoAccumulationRate*JTJ(i,i));
//                     }
//                 }

//                 float errPerObsPoint = optimizer.getErrPerObsPoint(1,0);
//                 float errPerModPoint = optimizer.getErrPerModPoint(1,0);

//                 infoLog.Log(errPerObsPoint,errPerObsPoint+errPerModPoint,stabilityThreshold,resetInfoThreshold);

//                 for (int m=0; m<tracker.getNumModels(); ++m) {
//                     for (int i=0; i<tracker.getPose(m).getReducedArticulatedDimensions(); ++i) {
//                         *poseVars[m][i+6] = tracker.getPose(m).getReducedArticulation()[i];
//                     }
//                     dart::SE3 T_cm = tracker.getPose(m).getTransformModelToCamera();
//                     *poseVars[m][0] = T_cm.r0.w; T_cm.r0.w = 0;
//                     *poseVars[m][1] = T_cm.r1.w; T_cm.r1.w = 0;
//                     *poseVars[m][2] = T_cm.r2.w; T_cm.r2.w = 0;
//                     dart::se3 t_cm = dart::se3FromSE3(T_cm);
//                     *poseVars[m][3] = t_cm.p[3];
//                     *poseVars[m][4] = t_cm.p[4];
//                     *poseVars[m][5] = t_cm.p[5];
//                 }
                
//                 // for lcm joints
//                 // publish optimized pose
//                 lcm_robot_state.publish_estimate();
//                 // publish frame poses of reported and estimated model
//                 lcm_frame_pub.publish_frame_pose("leftWristPitch");
//             }

//         }

//         //////////////////////////////////////////////////////////////////////////////////////////////////////////
//         //                                                                                                      //
//         //                                     Render this frame                                                //
//         //                                                                                                      //
//         //////////////////////////////////////////////////////////////////////////////////////////////////////////

//         glClearColor (1.0, 1.0, 1.0, 1.0);
//         glShadeModel (GL_SMOOTH);
//         float4 lightPosition = make_float4(normalize(make_float3(-0.4405,-0.5357,-0.619)),0);
//         glLightfv(GL_LIGHT0, GL_POSITION, (float*)&lightPosition);

//         camDisp.ActivateScissorAndClear(camState);

//         glEnable(GL_DEPTH_TEST);
//         glEnable(GL_LIGHT0);
//         glEnable(GL_NORMALIZE);
//         glEnable(GL_LIGHTING);

//         camDisp.ActivateAndScissor(camState);

//         // for urdf
//         glPushMatrix();
//         //static pangolin::Var<bool> showAxes("ui.showAxes",true,true);
//         static pangolin::Var<bool> showAxes("ui.showAxes",false,true);
//         if(showAxes) {
//             // draw coordinates axes, x: red, y: green, z: blue
//             pangolin::glDrawAxis(0.5);
//             glColor3f(0,0,1);
//             pangolin::glDraw_z0(0.01, 10);
//         }

//         if (showCameraPose) {

//             glColor3f(0,0,0);
//             glPushMatrix();

// //            glRotatef(180,0,1,0);
// //            glutSolidCube(0.02);

// //            glTranslatef(0,0,-0.02);
// //            glutSolidCone(0.0125,0.02,10,1);

//             glPopMatrix();

//         }

//         glColor4ub(0xff,0xff,0xff,0xff);
//         if (showEstimatedPose) {

//             glEnable(GL_COLOR_MATERIAL);

//             glPushMatrix();

//             if (showVoxelized) {
//                 glColor3f(0.2,0.3,1.0);

//                 for (int m=0; m<tracker.getNumModels(); ++m) {
// //                for (int m=1; m<=1; m+=10) {
//                     tracker.updatePose(m);
//                     tracker.getModel(m).renderVoxels(levelSet);
//                 }
//             }
//             else{
//                 for (int m=0; m<tracker.getNumModels(); ++m) {
//                     tracker.updatePose(m);
//                     tracker.getModel(m).render();
//                 }
//             }

//             glPopMatrix();

//         }

//         if (showReported) {
//             glColor3ub(0xfa,0x85,0x7c);
//             glEnable(GL_COLOR_MATERIAL);
//             // render Valkyrie reported state as wireframe model, origin is the camera centre
//             val.setPose(val_pose);
//             val.renderWireframe();
//             // glColor3ub(0,0,0);
//             // glutSolidSphere(0.02,10,10);
//         }

//         glPointSize(1.0f);

//         if (showObsSdf) {
//             static pangolin::Var<float> levelSet("ui.levelSet",0,-10,10);

//             for (int m=0; m<tracker.getNumModels(); ++m) {

//                 glPushMatrix();
//                 dart::glMultSE3(tracker.getModel(m).getTransformModelToCamera());
//                 tracker.getModel(m).syncObsSdfDeviceToHost();
//                 dart::Grid3D<float> * obsSdf = tracker.getModel(m).getObsSdf();
//                 tracker.getModel(m).renderSdf(*obsSdf,levelSet);
//                 glPopMatrix();
//             }
//         }

//         if (showTrackedPoints) {

//             glPointSize(4.0f);
//             glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudVbo);
//             glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(float4),tracker.getHostVertMap(),GL_DYNAMIC_DRAW_ARB);

//             glEnableClientState(GL_VERTEX_ARRAY);
//             glDisableClientState(GL_NORMAL_ARRAY);
//             glVertexPointer(4, GL_FLOAT, 0, 0);

//             if(showPointColour)
//                 pointColoringObs = PointColoringRGB;
//             else if(pointColoringObs == PointColoringRGB)
//                 pointColoringObs = PointColoringNone;

//             switch (pointColoringObs) {
//             case PointColoringNone:
//                 glColor3f(0.25,0.25,0.25);
//                 glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudNormVbo);
//                 glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(float4),tracker.getHostNormMap(),GL_DYNAMIC_DRAW_ARB);

//                 glNormalPointer(GL_FLOAT, 4*sizeof(float), 0);
//                 glEnableClientState(GL_NORMAL_ARRAY);
//                 break;
//             case PointColoringRGB:
//                 glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
//                 glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(uchar3),depthSource->getColor(),GL_DYNAMIC_DRAW_ARB);
//                 glColorPointer(3,GL_UNSIGNED_BYTE, 0, 0);
//                 glEnableClientState(GL_COLOR_ARRAY);
//                 glDisable(GL_LIGHTING);
//                 break;
//             case PointColoringErr:
//                 {
//                     static float errorMin = 0.0;
//                     static float errorMax = 0.1;
//                     float * dErr;
//                     cudaMalloc(&dErr,depthWidth*depthHeight*sizeof(float));
//                     dart::imageSquare(dErr,tracker.getDeviceDebugErrorObsToMod(),depthWidth,depthHeight);
//                     dart::colorRampHeatMapUnsat(imgDepthSize.devicePtr(),dErr,depthWidth,depthHeight,errorMin,errorMax);
//                     cudaFree(dErr);
//                     imgDepthSize.syncDeviceToHost();
//                     glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
//                     glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(uchar3),imgDepthSize.hostPtr(),GL_DYNAMIC_DRAW_ARB);
//                     glColorPointer(3,GL_UNSIGNED_BYTE, 0, 0);
//                     glEnableClientState(GL_COLOR_ARRAY);
//                     glDisable(GL_LIGHTING);
//                 }
//                 break;
//             case PointColoringDA:
//                 {
//                     const int * dDebugDA = tracker.getDeviceDebugDataAssociationObsToMod();
//                     dart::colorDataAssociationMultiModel(imgDepthSize.devicePtr(),dDebugDA,allSdfColors.devicePtr(),depthWidth,depthHeight);
//                     imgDepthSize.syncDeviceToHost();
//                     glBindBufferARB(GL_ARRAY_BUFFER_ARB,pointCloudColorVbo);
//                     glBufferDataARB(GL_ARRAY_BUFFER_ARB,depthWidth*depthHeight*sizeof(uchar3),imgDepthSize.hostPtr(),GL_DYNAMIC_DRAW_ARB);
//                     glColorPointer(3,GL_UNSIGNED_BYTE, 0, 0);
//                     glEnableClientState(GL_COLOR_ARRAY);
//                     glDisable(GL_LIGHTING);
//                 }
//                 break;
//             }

//             glDrawArrays(GL_POINTS,0,depthWidth*depthHeight);
//             glBindBuffer(GL_ARRAY_BUFFER_ARB,0);

//             glDisableClientState(GL_VERTEX_ARRAY);
//             glDisableClientState(GL_COLOR_ARRAY);
//             glDisableClientState(GL_NORMAL_ARRAY);

//             glPointSize(1.0f);

//         }

//         if (showPredictedPoints) {

//             glPointSize(4.0f);

//             const float4 * dPredictedVertMap = tracker.getDevicePredictedVertMap();
//             const int nPoints = tracker.getPredictionWidth()*tracker.getPredictionHeight();
//             float4 * hPredictedVertMap = new float4[nPoints];
//             cudaMemcpy(hPredictedVertMap,dPredictedVertMap,nPoints*sizeof(float4),cudaMemcpyDeviceToHost);


//             glDisable(GL_LIGHTING);
//             glBegin(GL_POINTS);

//             if (pointColoringPred == PointColoringErr) {
//                 static pangolin::Var<float> errMin("ui.errMin",0,0,0.05);
//                 static pangolin::Var<float> errMax("ui.errMax",0.01,0,0.05);
//                 dart::MirroredVector<uchar3> debugImg(tracker.getPredictionWidth()*tracker.getPredictionHeight());
//                 dart::colorRampHeatMapUnsat(debugImg.devicePtr(),
//                                             tracker.getDeviceDebugErrorModToObs(),
//                                             depthWidth,depthHeight,
//                                             errMin,errMax);
//                 debugImg.syncDeviceToHost();

//                 for (int i=0; i<nPoints; ++i) {
//                     if (hPredictedVertMap[i].z > 0) {
//                         uchar3 color = debugImg.hostPtr()[i];
//                         glColor3ubv((unsigned char*)&color);
//                         glVertex3f(hPredictedVertMap[i].x,hPredictedVertMap[i].y,hPredictedVertMap[i].z);
//                     }
//                 }

//             } else {

//                 for (int i=0; i<nPoints; ++i) {
//                     if (hPredictedVertMap[i].z > 0) {
//                         int id = round(hPredictedVertMap[i].w);
//                         int model = id >> 16;
//                         int sdf = id & 65535;
//                         uchar3 color = tracker.getModel(model).getSdfColor(sdf);
//                         glColor3ubv((unsigned char*)&color);
//                         glVertex3f(hPredictedVertMap[i].x,hPredictedVertMap[i].y,hPredictedVertMap[i].z);
//                     }
//                 }
//             }

//             glEnd();
//             delete [] hPredictedVertMap;

//             glPointSize(1.0f);
//         }

//         if (showCollisionClouds) {
//             glPointSize(10);
//             glColor3f(0,0,1.0f);
//             glDisable(GL_LIGHTING);
//             glBegin(GL_POINTS);
//             for (int m=0; m<tracker.getNumModels(); ++m) {
//                 const float4 * collisionCloud = tracker.getCollisionCloud(m);
//                 for (int i=0; i<tracker.getCollisionCloudSize(m); ++i) {
//                     int grid = round(collisionCloud[i].w);
//                     int frame = tracker.getModel(m).getSdfFrameNumber(grid);
//                     float4 v = tracker.getModel(m).getTransformModelToCamera()*
//                                tracker.getModel(m).getTransformFrameToModel(frame)*
//                                make_float4(make_float3(collisionCloud[i]),1.0);
//                     glVertex3fv((float*)&v);
//                 }
//             }
//             glEnd();
//             glEnable(GL_LIGHTING);

//             glPointSize(1);
//             glColor3f(1,1,1);
//         }

//         glPopMatrix();

//         imgDisp.ActivateScissorAndClear();
//         glDisable(GL_LIGHTING);
//         glColor4ub(255,255,255,255);

//         switch (debugImg) {
//             case DebugColor:
//             {
//                 if (depthSource->hasColor()) {
//                     imgTexDepthSize.Upload(depthSource->getColor(),GL_RGB,GL_UNSIGNED_BYTE);
//                     imgTexDepthSize.RenderToViewport();
//                 }
//             }
//             break;
//         case DebugObsDepth:
//             {

//                 static const float depthMin = 0.3;
//                 static const float depthMax = 1.0;

//                 //const unsigned short * depth = depthSource->getDepth();
//                 const auto * depth = depthSource->getDepth();

//                 for (int i=0; i<depthSource->getDepthWidth()*depthSource->getDepthHeight(); ++i) {
//                     if (depth[i] == 0) {
//                         imgDepthSize[i] = make_uchar3(128,0,0);
//                     } else {
//                         unsigned char g = std::max(0,std::min((int)(255*(depth[i]*depthSource->getScaleToMeters()-depthMin)/(float)(depthMax - depthMin)),255));
//                         imgDepthSize[i] = make_uchar3(g,g,g);
//                     }

//                 }

//                 imgTexDepthSize.Upload(imgDepthSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
//                 imgTexDepthSize.RenderToViewport();
//             }
//         case DebugPredictedDepth:
//             {
//                 static const float depthMin = 0.3;
//                 static const float depthMax = 1.0;

//                 const float4 * dPredictedVertMap = tracker.getDevicePredictedVertMap();
//                 static std::vector<float4> hPredictedVertMap(predWidth*predHeight);

//                 cudaMemcpy(hPredictedVertMap.data(),dPredictedVertMap,predWidth*predHeight*sizeof(float4),cudaMemcpyDeviceToHost);

//                 for (int i=0; i<predHeight*predWidth; ++i) {
//                     const float depth = hPredictedVertMap[i].z;
//                     if (depth == 0) {
//                         imgPredSize[i] = make_uchar3(128,0,0);
//                     } else {
//                         unsigned char g = std::max(0,std::min((int)(255*(depth-depthMin)/(float)(depthMax - depthMin)),255));
//                         imgPredSize[i] = make_uchar3(g,g,g);
//                     }
//                 }

//                 imgTexPredictionSize.Upload(imgPredSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
//                 imgTexPredictionSize.RenderToViewport();
//             }
//             break;
//         case DebugObsToModDA:
//         {
//             dart::colorDataAssociationMultiModel(imgDepthSize.devicePtr(),
//                                                  tracker.getDeviceDebugDataAssociationObsToMod(),
//                                                  allSdfColors.devicePtr(),depthWidth,depthHeight);\
//             imgDepthSize.syncDeviceToHost();
//             imgTexDepthSize.Upload(imgDepthSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
//             imgTexDepthSize.RenderToViewport();
//             break;
//         }
//         case DebugModToObsDA:
//         {
//             dart::colorDataAssociationMultiModel(imgPredSize.devicePtr(),
//                                                  tracker.getDeviceDebugDataAssociationModToObs(),
//                                                  allSdfColors.devicePtr(),predWidth,predHeight);\
//             imgPredSize.syncDeviceToHost();
//             imgTexPredictionSize.Upload(imgPredSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
//             imgTexPredictionSize.RenderToViewport();
//             break;
//         }
//         case DebugObsToModErr:
//             {
//                 static const float errMax = 0.01;
//                 dart::colorRampHeatMapUnsat(imgDepthSize.devicePtr(),
//                                             tracker.getDeviceDebugErrorObsToMod(),
//                                             depthWidth,depthHeight,
//                                             0.f,errMax);
//                 imgDepthSize.syncDeviceToHost();
//                 imgTexDepthSize.Upload(imgDepthSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
//                 imgTexDepthSize.RenderToViewport();
//             }
//             break;
//         case DebugModToObsErr:
//             {
//                 static const float errMax = 0.01;
//                 dart::colorRampHeatMapUnsat(imgPredSize.devicePtr(),
//                                             tracker.getDeviceDebugErrorModToObs(),
//                                             depthWidth,depthHeight,
//                                             0.f,errMax);
//                 imgPredSize.syncDeviceToHost();
//                 imgTexPredictionSize.Upload(imgPredSize.hostPtr(),GL_RGB,GL_UNSIGNED_BYTE);
//                 imgTexPredictionSize.RenderToViewport();
//             }
//             break;
//         case DebugJTJ:
//             imgTexDepthSize.Upload(tracker.getOptimizer()->getJTJimg(),GL_RGB,GL_UNSIGNED_BYTE);
//             imgTexDepthSize.RenderToViewportFlipY();
//             break;
//         default:
//             break;
//         }

//         cudaError_t err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             std::cerr << cudaGetErrorString(err) << std::endl;
//         }

//         if(pangolin::Pushed(record)) {
//                 pangolin::DisplayBase().RecordOnRender("ffmpeg:[fps=50,bps=8388608,unique_filename]//screencap.avi");
//         }

//         pangolin::FinishFrame();

//         if (pangolin::Pushed(stepVideo) || trackFromVideo || pangolinFrame == 1) {

//         } else {

//         }

//     }

//     glDeleteBuffersARB(1,&pointCloudVbo);
//     glDeleteBuffersARB(1,&pointCloudColorVbo);
//     glDeleteBuffersARB(1,&pointCloudNormVbo);

//     for (int m=0; m<tracker.getNumModels(); ++m) {
//         for (int i=0; i<tracker.getPose(m).getReducedDimensions(); ++i) {
//             delete poseVars[m][i];
//         }
//         delete [] poseVars[m];
//     }

//     for (uint i=0; i<sizeVars.size(); ++i) {
//         delete sizeVars[i];
//     }

//     delete depthSource;

    return 0;
}
