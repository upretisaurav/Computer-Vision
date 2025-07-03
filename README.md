# Computer Vision Learning Project

A comprehensive computer vision learning project that covers fundamental concepts, face detection/recognition, behavior tracking, and image processing techniques using OpenCV, MediaPipe, and other computer vision libraries.

## Project Structure

```
computer-vision/
├── README.md
├── requirements.txt
├── basic_face_detection/          # Face detection fundamentals
├── basic_image_processing_concepts/ # Core image processing
├── behaviour_tracking/            # Behavior analysis using facial landmarks
├── edge_detection/               # Edge detection algorithms
└── face_recognition/             # Face recognition and database management
```

## Features

### Face Detection & Recognition

- **Basic Face Detection**: Learn Haar Cascade-based face detection
- **Face Encoding**: Convert faces to 128-dimensional numerical representations
- **Face Matching**: Compare face encodings with configurable thresholds
- **Persistent Face Database**: Store and manage face data with metadata
- **Real-time Recognition**: Live face recognition from camera feed

### Behavior Tracking

- **Sleep Detection**: Monitor eye closure patterns to detect drowsiness
- **Yawn Detection**: Analyze mouth opening patterns to identify yawning
- **Real-time Analytics**: Track behavior episodes with timing statistics
- **Debug Mode**: Detailed logging and visualization for development

### Image Processing Fundamentals

- **Edge Detection**: Canny and Sobel edge detection algorithms
- **Image Filtering**: Gaussian blur and noise reduction
- **Morphological Operations**: Erosion, dilation, opening, closing
- **Feature Detection**: Corner detection (Harris, Shi-Tomasi, FAST)
- **Pattern Matching**: SIFT/ORB descriptors and feature matching

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- `opencv-python` - Computer vision operations
- `face-recognition` - Face detection and encoding
- `mediapipe` - Facial landmark detection
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `dlib` - Machine learning algorithms

## Getting Started

### 1. Face Detection (Beginner)

Learn the basics of detecting faces in images and video:

```bash
cd basic_face_detection
python simple_test.py
```

**What you'll learn:**

- How Haar Cascades work
- Parameter tuning for detection accuracy
- Real-time face detection from camera

### 2. Face Recognition (Intermediate)

#### Step 1: Understanding Face Encodings

```bash
cd face_recognition
python face_encoding.py
```

- Learn how faces are converted to numerical vectors
- Understand the 128-dimensional face representation

#### Step 2: Face Matching

```bash
python face_recognition_no_database.py
```

- Compare face encodings
- Understand similarity thresholds
- Build a temporary face matching system

#### Step 3: Persistent Database

```bash
python face_recognition_database.py
```

- Create a permanent face database
- Add, remove, and manage face records
- Includes metadata storage and retrieval

### 3. Behavior Tracking (Advanced)

Monitor human behavior using facial landmarks:

```bash
cd behaviour_tracking
python sleep_and_yawn_detection.py
```

**Features:**

- Real-time sleep detection based on eye closure
- Yawn detection using mouth aspect ratio
- Episode tracking with timing statistics
- Debug mode with detailed logging

### 4. Image Processing Concepts

Explore fundamental image processing techniques:

```bash
cd basic_image_processing_concepts

# Edge detection
python edge_detection/canny_edge_detection_basics.py

# Morphological operations
python morphological_operations/erosion.py
python morphological_operations/dilation.py

# Pattern matching
python pattern_matching/corner_detection.py
python pattern_matching/feature_matching.py
```

## 💡 Learning Path

### Beginner Track

1. **Face Detection** ([`basic_face_detection/simple_test.py`](basic_face_detection/simple_test.py))

   - Understand what face detection is
   - Learn about Haar Cascades
   - Practice parameter tuning

2. **Basic Image Processing** ([`basic_image_processing_concepts/`](basic_image_processing_concepts/))
   - Image filtering and noise reduction
   - Edge detection fundamentals
   - Morphological operations

### Intermediate Track

3. **Face Encoding** ([`face_recognition/face_encoding.py`](face_recognition/face_encoding.py))

   - Convert faces to numerical representations
   - Understand feature vectors

4. **Face Matching** ([`face_recognition/face_recognition_no_database.py`](face_recognition/face_recognition_no_database.py))
   - Compare face encodings
   - Threshold-based matching
   - Handle false positives/negatives

### Advanced Track

5. **Face Database** ([`face_recognition/face_recognition_database.py`](face_recognition/face_recognition_database.py))

   - Persistent storage using [`FaceDatabase`](face_recognition/face_recognition_database.py) class
   - Metadata management
   - Database operations (add, remove, search)

6. **Behavior Analysis** ([`behaviour_tracking/sleep_and_yawn_detection.py`](behaviour_tracking/sleep_and_yawn_detection.py))
   - MediaPipe facial landmarks
   - Real-time behavior monitoring
   - Statistical analysis

## Database Structure

The face recognition system uses two files for data persistence:

### Face Database Files

- **`face_encodings.pkl`** - Binary file storing face encoding vectors
- **[`face_metadata.json`](face_recognition/face_database/face_metadata.json)** - Human-readable metadata

Example metadata structure:

```json
{
  "person_name": {
    "added_data": "2025-06-26T13:23:54.048323",
    "face_id": 0,
    "additional_info": {
      "role": "developer"
    }
  }
}
```

## Interactive Features

### Face Recognition Database

- **Add faces** with custom names and metadata
- **Remove faces** from the database
- **Real-time recognition** with confidence scores
- **Database statistics** and management

### Behavior Tracker

- **Live monitoring** of sleep and yawn patterns
- **Configurable thresholds** for detection sensitivity
- **Episode tracking** with timing statistics
- **Reset functionality** for testing

### Parameter Tuning

- **Interactive sliders** for detection parameters
- **Real-time feedback** on parameter changes
- **Visual debugging** information

## Technical Details

### Face Recognition Pipeline

1. **Detection** - Locate faces in images using Haar Cascades
2. **Encoding** - Convert faces to 128D vectors using dlib's face recognition model
3. **Matching** - Compare encodings using Euclidean distance
4. **Threshold** - Apply similarity threshold (typically 0.6)

### Behavior Detection Algorithms

- **Eye Aspect Ratio (EAR)** - Monitor eye closure patterns
- **Mouth Aspect Ratio (MAR)** - Detect yawning behavior
- **Temporal Filtering** - Use frame counts to reduce false positives

### Performance Optimizations

- **Grayscale conversion** for faster processing
- **Image preprocessing** for better detection accuracy
- **Efficient data structures** for face database operations

## Troubleshooting

### Common Issues

1. **Camera not detected**

   ```python
   # Try different camera indices
   cap = cv2.VideoCapture(1)  # Instead of 0
   ```

2. **dlib installation issues**

   ```bash
   # On macOS
   brew install cmake
   pip install dlib

   # On Ubuntu
   sudo apt-get install cmake
   pip install dlib
   ```

3. **Face recognition accuracy**
   - Ensure good lighting conditions
   - Position face clearly in camera
   - Adjust detection thresholds if needed

## Learning Objectives

By completing this project, you will understand:

- **Computer Vision Fundamentals**

  - Image processing concepts
  - Feature detection and description
  - Pattern matching algorithms

- **Face Recognition Technology**

  - Detection vs Recognition distinction
  - Face encoding and matching
  - Database management for biometric systems

- **Real-time Processing**

  - Video stream handling
  - Performance optimization
  - User interface design

- **Machine Learning Applications**
  - Pre-trained models usage
  - Threshold tuning
  - False positive/negative handling

## Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [dlib Documentation](http://dlib.net/)
- [Face Recognition Library](https://face-recognition.readthedocs.io/)

## Contributing

This is a learning project. Feel free to:

- Add new computer vision examples
- Improve documentation
- Optimize algorithms
- Add new behavior detection features

## License

This project is for educational purposes. Please ensure you have appropriate permissions when using face recognition technology in production environments.

---

**Happy Learning! 🎉**

Start with the basic face detection examples and gradually work your way up to the advanced behavior tracking features. Each module is designed to build upon the previous concepts while introducing new computer vision techniques.
