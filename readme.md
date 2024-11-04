# 3D Object Detection with MediaPipe Objectron

## Overview
This project implements real-time 3D object detection using MediaPipe Objectron and OpenCV. The system can detect and track objects in 3D space through webcam feed, providing bounding boxes and orientation axes for detected objects.

## Features
- Real-time 3D object detection
- Webcam integration 
- 3D bounding box visualization
- Object orientation tracking
- Support for multiple object categories

## Prerequisites
Before running this project, ensure you have installed:
- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install required packages:
```bash
pip install opencv-python mediapipe numpy
```

## Usage
Run the detection script:
```bash
python object_detection.py
```

### Controls
- Press 'ESC' to exit the application

### Supported Objects
The following objects can be detected:
- Cups
- Chairs
- Shoes
- Cameras

## MediaPipe Pros and Cons

### Advantages
1. **Easy Integration**
   - Simple API
   - Minimal code required
   - Cross-platform compatibility

2. **Performance**
   - Real-time processing
   - GPU acceleration support
   - Optimized for mobile devices

3. **Features**
   - Pre-trained models available
   - Multiple object category support
   - Accurate 3D orientation estimation

4. **Development**
   - Active community support
   - Regular updates and improvements
   - Comprehensive documentation

### Disadvantages
1. **Limitations**
   - Limited object categories
   - Fixed model architectures
   - Dependency on good lighting conditions

2. **Resource Usage**
   - High CPU/GPU usage
   - Memory intensive
   - Battery drain on mobile devices

3. **Customization**
   - Limited model customization
   - Difficult to train custom objects
   - Fixed detection parameters

4. **Technical Constraints**
   - Requires modern hardware
   - Internet connection for initial setup
   - Version compatibility issues

## Configuration
Adjust detection parameters in the code:
```python
objectron = mp_objectron.Objectron(
    static_image_mode=False,
    max_num_objects=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.99,
    model_name='Cup'
)
```

## Troubleshooting
Common issues and solutions:
1. **No Detection**
   - Check lighting conditions
   - Verify camera connection
   - Ensure object is within frame

2. **Poor Performance**
   - Reduce max_num_objects
   - Lower resolution input
   - Close background applications

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Submit pull request

## License
This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments
- MediaPipe team for the Objectron model
- OpenCV community
- All contributors to this project
