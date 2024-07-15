#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/Dense> // For Eigen::Vector4f
#include <pcl/common/common.h>
#include <thread>  // for std::this_thread::sleep_for
#include <chrono>  // for std::chrono::milliseconds
#include <iostream>

struct Color
{

	float r, g, b;

	Color(float setR, float setG, float setB)
		: r(setR), g(setG), b(setB)
	{}
};

struct Box
{
	float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
};

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr loadPcd(std::string file)
{
    /**
     * @brief Loads a Point Cloud Data (PCD) file into a PointCloud object.
     * 
     * @param file The file path of the PCD file to load.
     * @return A shared pointer to the loaded PointCloud object if successful, nullptr otherwise.
     */
    
    // Create a new PointCloud object of type PointT
    typename pcl::PointCloud<PointT>::Ptr cloud(new typename pcl::PointCloud<PointT>);
    
    // Try to load the PCD file into the PointCloud object
    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1)
    {
        // Print an error message if the file could not be loaded
        PCL_ERROR("Couldn't read file \n");
        // Return nullptr to indicate failure
        return nullptr;
    }
    
    // Print the number of points loaded from the file
    std::cout << "Loaded " << cloud->points.size() << " data points from " + file << std::endl;
    
    // Return the loaded PointCloud object
    return cloud;
}

void renderPointCloud(pcl::visualization::PCLVisualizer::Ptr& viewer, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const std::string& name, Color color)
{
    /**
     * @brief Renders a PointCloud object in the PCLVisualizer.
     * 
     * @param viewer A pointer to the PCLVisualizer object.
     * @param cloud A shared pointer to the PointCloud object to render.
     * @param name The name identifier for the rendered PointCloud.
     * @param color The color for rendering the PointCloud.
     */
    
    // Check if the color is set to a default value indicating the use of intensity for coloring
    if (color.r == -1)
    {
        // Create a color handler that colors points based on their intensity
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(cloud, "intensity");
        // Add the point cloud to the viewer with the intensity color handler
        viewer->addPointCloud<pcl::PointXYZI>(cloud, intensity_distribution, name);
    }
    else
    {
        // Add the point cloud to the viewer with a default color handler
        viewer->addPointCloud<pcl::PointXYZI>(cloud, name);
        // Set the color of the point cloud to the specified RGB values
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, name);
    }
    
    // Set the point size property for the point cloud
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{
    /**
     * @brief Filters a PointCloud object using a crop box and voxel grid filter.
     * 
     * @param cloud A shared pointer to the PointCloud object to filter.
     * @param filterRes The voxel grid leaf size for downsampling.
     * @param minPoint The minimum XYZ coordinates defining the crop box.
     * @param maxPoint The maximum XYZ coordinates defining the crop box.
     * @return A shared pointer to the filtered PointCloud object.
     */
    
    // Create a new PointCloud object to store the result of the crop box filter
    typename pcl::PointCloud<PointT>::Ptr cloud_box(new pcl::PointCloud<PointT>());
    
    // Create and configure the crop box filter
    typename pcl::CropBox<PointT> cropBoxFilter(true);
    cropBoxFilter.setInputCloud(cloud);
    cropBoxFilter.setMin(minPoint);
    cropBoxFilter.setMax(maxPoint);
    
    // Apply the crop box filter
    cropBoxFilter.filter(*cloud_box);

    // Create a new PointCloud object to store the result of the voxel grid filter
    typename pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    
    // Create and configure the voxel grid filter
    typename pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud_box);
    sor.setLeafSize(filterRes, filterRes, filterRes);
    
    // Apply the voxel grid filter
    sor.filter(*cloud_filtered);

    // Create a vector to store indices of points that are in the roof region
    std::vector<int> indices;
    
    // Create and configure another crop box filter for the roof region
    typename pcl::CropBox<PointT> roofFilter(true);
    roofFilter.setInputCloud(cloud_filtered);
    roofFilter.setMin(Eigen::Vector4f(-1.5, -1.7, -1.1, 1.0));
    roofFilter.setMax(Eigen::Vector4f(2.6, 1.7, -0.4, 1.0));
    
    // Apply the roof filter to get the indices of points within the roof region
    roofFilter.filter(indices);

    // Create a PointIndices object to store the indices
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    for (int point : indices)
    {
        inliers->indices.push_back(point);
    }
    
    // Create a new PointCloud object to store the final filtered output
    typename pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>());
    
    // Create and configure the extract indices filter
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(true); // Remove the points in the roof region
    
    // Apply the extract indices filter
    extract.filter(*cloud_out);

    // Return the filtered point cloud
    return cloud_out;
}


template <typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    /**
     * @brief Segments a point cloud into two parts: one containing the planar surface and the other containing the remaining points.
     * 
     * @param cloud A shared pointer to the input PointCloud.
     * @param maxIterations The maximum number of iterations for the RANSAC algorithm.
     * @param distanceThreshold The distance threshold for considering a point to be an inlier.
     * @return A pair of shared pointers to the segmented PointClouds: the first is the cloud with the remaining points, and the second is the cloud with the planar surface points.
    */
    // Create the segmentation object
    typename pcl::SACSegmentation<PointT> seg;
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    // Set segmentation parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);
    seg.setInputCloud(cloud);

    // Segment the largest planar component from the input cloud
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0)
    {
        std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    // Create point clouds to hold the segmented points
    typename pcl::PointCloud<PointT>::Ptr obstCloud (new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr planeCloud (new pcl::PointCloud<PointT>());

    // Separate the points in the plane
    for (auto point : inliers->indices)
    {
        planeCloud->points.push_back(cloud->points[point]);
    }

    // Create the filtering object
    typename pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstCloud);

    // Create a pair to store the segmented clouds
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);

    // Return the result
    return segResult;
}

template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    /**
     * @brief Clusters a point cloud using Euclidean cluster extraction.
     * 
     * @param cloud A shared pointer to the input PointCloud.
     * @param clusterTolerance The spatial cluster tolerance as a measure in the L2 Euclidean space.
     * @param minSize The minimum number of points that a cluster needs to contain.
     * @param maxSize The maximum number of points that a cluster needs to contain.
     * @return A vector of shared pointers to the clustered PointClouds.
     */
    // Vector to store the resulting clusters
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // Create a KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    // Vector to store the indices of the points in each cluster
    std::vector<pcl::PointIndices> cluster_indices;

    // Create the Euclidean cluster extraction object
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // Iterate through the indices of each cluster
    for (auto& indices : cluster_indices)
    {
        typename pcl::PointCloud<PointT>::Ptr clusterCloud(new pcl::PointCloud<PointT>());

        // Add the points corresponding to the current cluster to clusterCloud
        for (auto index : indices.indices)
            clusterCloud->points.push_back(cloud->points[index]);

        // Set the properties of the cluster cloud
        clusterCloud->width = clusterCloud->points.size();
        clusterCloud->height = 1;
        clusterCloud->is_dense = true;

        // Add the cluster cloud to the list of clusters
        clusters.push_back(clusterCloud);
    }

    return clusters;
}

template <typename PointT>
Box BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{
    PointT minPoint,maxPoint;
    pcl::getMinMax3D(*cluster,minPoint,maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;

}

// Draw wire frame box with filled transparent color 
void renderBox(pcl::visualization::PCLVisualizer::Ptr& viewer, Box box, int id, Color color, float opacity)
{
	if(opacity > 1.0)
		opacity = 1.0;
	if(opacity < 0.0)
		opacity = 0.0;
	
	std::string cube = "box"+std::to_string(id);
    //viewer->addCube(box.bboxTransform, box.bboxQuaternion, box.cube_length, box.cube_width, box.cube_height, cube);
    viewer->addCube(box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max, color.r, color.g, color.b, cube);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, cube); 
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, cube);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, cube);
    
    std::string cubeFill = "boxFill"+std::to_string(id);
    //viewer->addCube(box.bboxTransform, box.bboxQuaternion, box.cube_length, box.cube_width, box.cube_height, cubeFill);
    viewer->addCube(box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max, color.r, color.g, color.b, cubeFill);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, cubeFill); 
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, cubeFill);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity*0.3, cubeFill);
}

std::vector<boost::filesystem::path> streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}

void single_pcd(pcl::visualization::PCLVisualizer::Ptr& viewer,pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZI>);
    filteredCloud = FilterCloud<pcl::PointXYZI>(inputCloud, 0.2, Eigen::Vector4f(-15, -6.0, -3, 1), Eigen::Vector4f(30, 6.0, 10, 1));

    // renderPointCloud(viewer, filteredCloud, "filteredCloud",Color(-1, -1, -1));

    std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> segmentCloud = SegmentPlane<pcl::PointXYZI>(filteredCloud,100,0.2);
    // renderPointCloud(viewer, segmentCloud.first, "obstCloud",Color(1,0,0));
    renderPointCloud(viewer, segmentCloud.second, "planeCloud",Color(0,1,0));

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters = Clustering<pcl::PointXYZI>(segmentCloud.first, 0.5, 10, 600);
    int clusterId = 0;
    std::vector<Color> colors = {Color(1,0,0),Color(0,0,1),Color(0,1,1)};
    for (auto cluster: cloudClusters)
    {
        renderPointCloud(viewer, cluster, "obstCloud"+std::to_string(clusterId),colors[clusterId %(colors.size())]);
        Box box = BoundingBox<pcl::PointXYZI>(cluster);
        renderBox(viewer,box,clusterId,Color(0.5,0.5,0.5),0.5);
        ++clusterId;

    }
    
}

enum CameraAngle
{
	XY, TopDown, Side, FPS
};

//setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    viewer->setBackgroundColor (0, 0, 0);
    
    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 16;
    
    switch(setAngle)
    {
        case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
        case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }

    if(setAngle!=FPS)
        viewer->addCoordinateSystem (1.0);
}






int main()
{
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZI>);
    

    
    CameraAngle setAngle = XY;
    initCamera(setAngle, viewer);

    // inputCloud = loadPcd<pcl::PointXYZI>("/home/arun/pointcloud_handling/0000000000.pcd");
    // single_pcd(viewer,inputCloud);
    // while (!viewer->wasStopped())
    // {
    //     viewer->spinOnce();
    // }
    
    std::vector<boost::filesystem::path> stream = streamPcd("../data/");
    auto streamIterator = stream.begin();

    // Define the desired FPS
    int fps = 30;
    int delay = 1000 / fps;  // Delay in milliseconds

    while (!viewer->wasStopped ())
    {

    // Clear viewer
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    // Load pcd and run obstacle detection process
    inputCloud = loadPcd<pcl::PointXYZI>((*streamIterator).string());
    single_pcd(viewer, inputCloud);        
    streamIterator++;

    if(streamIterator == stream.end())
        streamIterator = stream.begin();

    viewer->spinOnce ();
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }

    return 0;
}



