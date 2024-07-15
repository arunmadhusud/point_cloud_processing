# Point Cloud Processing with PCL


This project showcases a range of point cloud processing techniques using the Point Cloud Library (PCL). It focuses on four key operations essential for understanding and analyzing 3D sensor data, particularly in the context of autonomous vehicles:

1. Downsampling & Filtering: Reducing point cloud density while preserving important features.
2. Segmentation: Separating the ground plane from objects of interest.
3. Clustering: Grouping points that likely belong to the same object.
4. Bounding Box Creation: Encapsulating clustered objects further analysis.

The project utilizes few point cloud files from a self-driving car dataset. The datset was provided by Udaicty as part of their Sensor Fusion Nanodegree program.

## Installation

To run this project, you'll need to have PCL and its dependencies installed. Follow these steps:

1. **Install PCL and its dependencies:**

    For Ubuntu:
    ```bash
    sudo apt update
    sudo apt install libpcl-dev
    ```

2. Clone the repository:
    ```bash
    git clone 
    cd point-cloud-processing
    ```

3. Create a build directory and compile the project:
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```


## Usage

After installing the necessary dependencies, you can run the project scripts to perform point cloud processing tasks.

```bash
./point_cloud_processing
```

## Results
- Output point cloud stream


```markdown
![Alt Text](/home/arun/pointcloud_handling/output.gif)
```

## Acknowledgements

- [Point Cloud Library (PCL)](https://pointclouds.org/)
- [Udacity Sensor Fusion Nanodegree](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313)
