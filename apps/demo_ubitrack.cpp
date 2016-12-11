#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <kfusion/kinfu.hpp>
#include <kfusion/cuda/marching_cubes.hpp>
#include <fstream>
#include <iostream>

#include <functional>
#include <condition_variable>
#include <mutex>


#include <utFacade/BasicFacadeTypes.h>
#include <utFacade/BasicFacade.h>

using namespace kfusion;
using namespace Ubitrack;




typedef unsigned long long int TimestampT;

class UTConnector {
public:
    UTConnector(const char* _components_path)
    : m_utFacade( _components_path )
    , m_haveNewFrame( false )
    , m_lastTimestamp( 0 )
    , m_dataflowLoaded( false )
    , m_dataflowRunning( false )
    , m_pushsink_camera_depth( NULL )
    , m_pullsink_camera_rgb( NULL )
    {
    }

    ~UTConnector() {
        // what needs to be done for teardown ??
    }

    /*
     * waits for an image to be pushed (left-eye) and returns its timestamp
     */
    TimestampT wait_for_frame() {
        // find a way to exit here in case we want to stop
        // alternatively, we could only query the m_haveNewFrame variable (polling)
        unsigned long long int ts(0);
        while (!m_haveNewFrame) {
            std::unique_lock<std::mutex>  ul( m_waitMutex );
            m_waitCondition.wait( ul );
            ts = m_lastTimestamp;
        }

        // reset haveNewFrame immediately to prepare for the next frame
        // maybe this should be done in a seperate method ??
        {
            std::unique_lock<std::mutex>  ul( m_waitMutex );
            m_haveNewFrame = false;
        }
        return ts;
    }


    /*
     * livecycle management for the utconnector
     * not thread-safe
     */
    virtual bool initialize(const char* _utql_filename) {
        try {
            m_utFacade.loadDataflow( _utql_filename );
            m_dataflowLoaded = true;

            if (m_pushsink_camera_depth != NULL) {
                delete m_pushsink_camera_depth;
            }
            m_pushsink_camera_depth = m_utFacade.getPushSink<Facade::BasicImageMeasurement>("depth_image");

            if(m_pullsink_camera_rgb != NULL)
            {
                delete m_pullsink_camera_rgb;
            }
            m_pullsink_camera_rgb = m_utFacade.getPullSink<Facade::BasicImageMeasurement>("rgb_image");

            if (m_pushsink_camera_depth) {
                m_pushsink_camera_depth->registerCallback(std::bind(&UTConnector::receive_depth_image, this, std::placeholders::_1));
            }

        }
        catch(std::exception & e)
        {
            std::cerr << "error initializing UTConnector: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    virtual bool teardown(){
        try {
            if (m_dataflowRunning) {
                stop();
            }

            if (m_pushsink_camera_depth != NULL) {
                m_pushsink_camera_depth->unregisterCallback();
                delete m_pushsink_camera_depth;
            }

            if(m_pullsink_camera_rgb != NULL)
            {
                delete m_pullsink_camera_rgb;
            }

            if (m_dataflowLoaded) {
                m_utFacade.clearDataflow();
                m_dataflowLoaded = false;
            }
        }
        catch(std::exception & e)
        {
            std::cerr << "error tearing down UTConnector: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    virtual bool start() {
        try {
            if (m_dataflowLoaded) {
                m_utFacade.startDataflow();
                m_dataflowRunning = true;
            }
        }
        catch(std::exception & e)
        {
            std::cerr << "error starting UTConnector: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    virtual bool stop() {
        try {
            if (m_dataflowRunning) {
                m_utFacade.stopDataflow();
                m_dataflowRunning = false;
            }
        }
        catch(std::exception & e)
        {
            std::cerr << "error stopping UTConnector: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

    inline TimestampT now() {
        return m_utFacade.now();
    }

    bool get_depth_image(const TimestampT ts, std::shared_ptr<Facade::BasicImageMeasurement> & img) {
        // we need locking here to prevent concurrent access to m_depth_image (when receiving new frame)
        if (ts != m_lastTimestamp) {
            return false;
        }

        {
            std::unique_lock<std::mutex> ul( m_textureAccessMutex );
            img = m_depth_image;
        }

        return true;
    }

    bool get_rgb_image(const TimestampT ts, std::shared_ptr<Facade::BasicImageMeasurement> & img) {
        if (m_pullsink_camera_rgb == NULL) {
            std::cerr << "pullsink depth left is not connected" << std::endl;
            return false;
        }

        try{
            std::shared_ptr<Facade::BasicImageMeasurement> m_image = m_pullsink_camera_rgb->get(ts);
            img = m_image;
        }
        catch(std::exception & e)
        {
            std::cerr << "error pulling camera image left: " << e.what() << std::endl;
            return false;
        }
        return true;

    }

    void receive_depth_image(std::shared_ptr<Facade::BasicImageMeasurement>& img) {
        std::cout << "got image" << std::endl;
        {
            std::unique_lock<std::mutex> ul( m_textureAccessMutex );
            m_depth_image = img;
        }
        // notify renderer that new frame is available
        set_new_frame(img->time());
    }


protected:

    void set_new_frame(TimestampT ts) {
        std::unique_lock<std::mutex> ul( m_waitMutex );
        m_lastTimestamp = ts;
        m_haveNewFrame = true;
        m_waitCondition.notify_one();
    }

    Facade::BasicFacade m_utFacade;
    bool m_haveNewFrame;
    TimestampT m_lastTimestamp;

    bool m_dataflowLoaded;
    bool m_dataflowRunning;

    Ubitrack::Facade::BasicPushSink< Facade::BasicImageMeasurement >*  m_pushsink_camera_depth;
    Ubitrack::Facade::BasicPullSink< Facade::BasicImageMeasurement >*  m_pullsink_camera_rgb;

    std::shared_ptr<Facade::BasicImageMeasurement > m_depth_image;


private:
    std::mutex m_waitMutex;
    std::mutex m_textureAccessMutex;
    std::condition_variable m_waitCondition;
};






struct KinFuApp
{
    void print_help()
    {
        std::cout << std::endl;
        std::cout << "kinfu_remake app hotkeys" << std::endl;
        std::cout << "=================" << std::endl;
        std::cout << "    H    : print VTK help" << std::endl;
        std::cout << "    K    : print this help" << std::endl;
        std::cout << "    Q    : exit" << std::endl;
        std::cout << "    T    : take cloud" << std::endl;
        std::cout << "    M    : take mesh and save as a .ply file" << std::endl;
        std::cout << "    I    : toggle iteractive mode" << std::endl;
        std::cout << "=================" << std::endl;
        std::cout << std::endl;
    } 

    static void KeyboardCallback(const cv::viz::KeyboardEvent& event, void* pthis)
    {
        KinFuApp& kinfu = *static_cast<KinFuApp*>(pthis);

        if(event.action != cv::viz::KeyboardEvent::KEY_DOWN)
            return;

        if(event.code == 't' || event.code == 'T')
            kinfu.take_cloud(*kinfu.kinfu_);

        if(event.code == 'i' || event.code == 'I')
            kinfu.iteractive_mode_ = !kinfu.iteractive_mode_;

        if(event.code == 'm' || event.code == 'M')
            kinfu.take_mesh(*kinfu.kinfu_);

        if(event.code == 's' || event.code == 'S')
            kinfu.write_mesh(*kinfu.kinfu_);

        if(event.code == 'k' || event.code == 'K')
            kinfu.print_help();
    }

    KinFuApp(UTConnector& connector) : exit_ (false),  iteractive_mode_(false), connector_ (connector), pause_(false)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        // need to reproject Depth to RGB Image here ??
        //capture_.setRegistration(true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    KinFuApp(UTConnector& connector, const KinFuParams& params) : exit_ (false),  iteractive_mode_(false), connector_ (connector), pause_(false)
    {
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        // need to reproject Depth to RGB Image here ??
        //capture_.setRegistration(true);

        cv::viz::WCube cube(cv::Vec3d::all(0), cv::Vec3d(params.volume_size), true, cv::viz::Color::apricot());
        viz.showWidget("cube", cube, params.volume_pose);
        viz.showWidget("coor", cv::viz::WCoordinateSystem(0.1));
        viz.registerKeyboardCallback(KeyboardCallback, this);
    }

    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/4000);
        cv::imshow("Depth", display);
    }

    void show_raycasted(KinFu& kinfu)
    {
        const int mode = 3;
        if (iteractive_mode_)
            kinfu.renderImage(view_device_, viz.getViewerPose(), mode);
        else
            kinfu.renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        cv::imshow("Scene", view_host_);
    }

    void take_cloud(KinFu& kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer_);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        if (kinfu.params().integrate_color) {
            kinfu.color_volume()->fetchColors(cloud, color_buffer_);
            cv::Mat color_host(1, (int)cloud.size(), CV_8UC4);
            cloud.download(cloud_host.ptr<Point>());
            color_buffer_.download(color_host.ptr<RGB>());
            viz.showWidget("cloud", cv::viz::WCloud(cloud_host, color_host));
        } else
        {
            cloud.download(cloud_host.ptr<Point>());
            viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        }
    }

    void write_mesh(KinFu& kinfu )
    {

        std::string outFilename = "mesh.ply";
        //first take mesh

        if (!marching_cubes_)
            marching_cubes_ = cv::Ptr<cuda::MarchingCubes>(new cuda::MarchingCubes());

        cuda::DeviceArray<Point> triangles = marching_cubes_->run(kinfu.tsdf(), triangles_buffer_);
        int n_vert = triangles.size();

        cv::viz::Mesh mesh;
        mesh.cloud.create(1, n_vert, CV_32FC4);
        mesh.polygons.create(1, 4*n_vert/3, CV_32SC1);

        for (int i = 0; i < n_vert/3; ++i) {
            mesh.polygons.at<int>(4*i) = 3;
            mesh.polygons.at<int>(4*i+1) = 3*i;
            mesh.polygons.at<int>(4*i+2) = 3*i+1;
            mesh.polygons.at<int>(4*i+3) = 3*i+2;
        }

        cv::Mat mesh_colors(1, n_vert, CV_8UC4);

        if (kinfu.params().integrate_color)
        {
            kinfu.color_volume()->fetchColors(triangles, color_buffer_);
            color_buffer_.download(mesh_colors.ptr<RGB>());
            mesh.colors = mesh_colors;
        }

        triangles.download(mesh.cloud.ptr<Point>());

        std::ofstream outFile( outFilename.c_str() );

        if ( !outFile )
        {
            std::cerr << "Error opening output file: " << outFilename << "!" << std::endl;
            exit( 1 );
        }

        ////
        // Header
        ////
        const int pointNum    =  mesh.cloud.cols;
        const int triangleNum =  mesh.polygons.cols/4; //polygons is a Mat where each column has only 1 value
        std::cout<<pointNum<<" "<<triangleNum<<std::endl;
        
        outFile << "ply" << std::endl;
        outFile << "format ascii 1.0" << std::endl;
        outFile << "element vertex " << pointNum << std::endl;
        outFile << "property float x" << std::endl;
        outFile << "property float y" << std::endl;
        outFile << "property float z" << std::endl;
        outFile << "property uchar red" << std::endl;
        outFile << "property uchar green" << std::endl;
        outFile << "property uchar blue" << std::endl;
        outFile << "element face " << triangleNum << std::endl;
        outFile << "property list uchar int vertex_index" << std::endl;
        outFile << "end_header" << std::endl;

        ////
        // Points and colors
        ////
        std::vector<cv::Mat> channels_cloud(4);
        std::vector<cv::Mat> channels_colors(4);
        split(mesh.cloud,channels_cloud);
        split(mesh.colors,channels_colors);
        for ( int i = 0; i < pointNum; i++ )
        {
            outFile << channels_cloud[0].at<float>(0,i)<<" ";       //x
            outFile << channels_cloud[1].at<float>(0,i)<<" ";       //y
            outFile << channels_cloud[2].at<float>(0,i)<<" ";       //z
            outFile << (int)channels_colors[2].at<uchar>(0,i)<<" "; //b
            outFile << (int)channels_colors[1].at<uchar>(0,i)<<" "; //g
            outFile << (int)channels_colors[0].at<uchar>(0,i)<<" "; //r

            outFile << std::endl;
        }

        ////
        // Triangles
        ////
        for ( int i = 0; i < triangleNum*4; i+=4 )
        {
            outFile << mesh.polygons.at<int>(0,i+0)<<" ";             
            outFile << mesh.polygons.at<int>(0,i+1)<<" ";             
            outFile << mesh.polygons.at<int>(0,i+2)<<" ";             
            outFile << mesh.polygons.at<int>(0,i+3)<<" ";             

            outFile << std::endl;
        }
    }

    void take_mesh(KinFu& kinfu)
    {
        if (!marching_cubes_)
            marching_cubes_ = cv::Ptr<cuda::MarchingCubes>(new cuda::MarchingCubes());

        cuda::DeviceArray<Point> triangles = marching_cubes_->run(kinfu.tsdf(), triangles_buffer_);
        int n_vert = triangles.size();

        cv::viz::Mesh mesh;
        mesh.cloud.create(1, n_vert, CV_32FC4);
        mesh.polygons.create(1, 4*n_vert/3, CV_32SC1);

        for (int i = 0; i < n_vert/3; ++i) {
            mesh.polygons.at<int>(4*i) = 3;
            mesh.polygons.at<int>(4*i+1) = 3*i;
            mesh.polygons.at<int>(4*i+2) = 3*i+1;
            mesh.polygons.at<int>(4*i+3) = 3*i+2;
        }

        cv::Mat mesh_colors(1, n_vert, CV_8UC4);

        if (kinfu.params().integrate_color)
        {
            kinfu.color_volume()->fetchColors(triangles, color_buffer_);
            color_buffer_.download(mesh_colors.ptr<RGB>());
            mesh.colors = mesh_colors;
        }

        triangles.download(mesh.cloud.ptr<Point>());

        

        viz.showWidget("cloud", cv::viz::WMesh(mesh));
    }

    bool execute()
    {
        KinFu& kinfu = *kinfu_;
        std::shared_ptr<Facade::BasicImageMeasurement> depth_img, rgb_img;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;

        for (int i = 0; !exit_ && !viz.wasStopped(); ++i)
        {
            TimestampT ts = connector_.wait_for_frame();
            connector_.get_depth_image(ts, depth_img);
            connector_.get_rgb_image(ts, rgb_img);

            depth = cv::Mat(cv::Size( depth_img->getDimX(), depth_img->getDimY()), 
                cv::Mat::MAGIC_VAL + CV_MAKE_TYPE(IPL2CV_DEPTH(depth_img->getPixelSize()), depth_img->getChannels()),
                depth_img->getDataPtr(), depth_img->getStep());
            image = cv::Mat(cv::Size( rgb_img->getDimX(), rgb_img->getDimY()), 
                cv::Mat::MAGIC_VAL + CV_MAKE_TYPE(IPL2CV_DEPTH(rgb_img->getPixelSize()), rgb_img->getChannels()),
                rgb_img->getDataPtr(), rgb_img->getStep());

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);
            color_device_.upload(image.data, image.step, image.rows, image.cols);

            {
                SampledScopeTime fps(time_ms); (void)fps;
                if (kinfu.params().integrate_color)
                    has_image = kinfu(depth_device_, color_device_);
                else
                    has_image = kinfu(depth_device_);
            }

            if (has_image)
                show_raycasted(kinfu);

            show_depth(depth);
            if (kinfu.params().integrate_color)
                cv::imshow("Image", image);

            if (!iteractive_mode_)
                viz.setViewerPose(kinfu.getCameraPose());

            int key = cv::waitKey(pause_ ? 0 : 3);
            
            switch(key)
            {
            case 't': case 'T' : take_cloud(kinfu); break;
            case 'i': case 'I' : iteractive_mode_ = !iteractive_mode_; break;
            case 'm': case 'M' : take_mesh(kinfu); break;
            case 27: exit_ = true; break;
            case 32: pause_ = !pause_; break;
            }

            //exit_ = exit_ || i > 100;
            viz.spinOnce(3, true);
        }
        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, iteractive_mode_;
    UTConnector& connector_;
    KinFu::Ptr kinfu_;
    cv::viz::Viz3d viz;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::Image color_device_;
    cuda::DeviceArray<Point> cloud_buffer_;
    cuda::DeviceArray<RGB> color_buffer_;
    cv::Ptr<cuda::MarchingCubes> marching_cubes_;
    cuda::DeviceArray<Point> triangles_buffer_;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
    cuda::setDevice (device);
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

    Facade::initUbitrackLogging("log4cpp.conf");

    UTConnector connector(argv[1]);
    connector.initialize (argv[2]);
   
    KinFuParams custom_params = KinFuParams::default_params();
    custom_params.integrate_color = true;

    KinFuApp app (connector, custom_params);

    // executing
    try { 
        connector.start();
        app.execute (); 
        connector.stop();
    }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    connector.teardown();
    
    return 0;
}
