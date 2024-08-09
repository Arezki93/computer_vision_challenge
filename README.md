# Real-Time Object Detection with SSD MobileNetV1 and TensorFlow Lite

## Introduction

This project demonstrates the implementation of a real-time object detection algorithm using the SSD MobileNetV1 model, optimized with TensorFlow Lite (TFLite) for deployment on portable devices. The goal is to develop a lightweight, fast, and accurate inference routine capable of processing video streams in real time.

## Model Selection

### Why SSD MobileNetV1?

The SSD MobileNetV1 model was chosen due to its balance between performance, accuracy, and resource efficiency. Here are the main reasons for this choice:

- **Portability**: The model is compact, with a size of approximately 3.99 MB, making it suitable for deployment on devices with limited memory and storage capacity.
- **Speed**: The model offers fast inference times, essential for real-time applications. It achieves an inference speed of 20 ms on a Pixel 4 device using the GPU and 29 ms on the CPU. This ensures the model can process video images quickly enough to provide real-time feedback.
- **Accuracy**: Although SSD MobileNetV1 is optimized for speed, it still offers a reasonable level of accuracy, with a COCO mAP of 21. This makes it suitable for applications where real-time processing is more critical than achieving the highest possible accuracy.

References:
- [TensorFlow Lite Object Detection Overview](https://www.tensorflow.org/lite/examples/object_detection/overview?hl=fr)
- [TensorFlow Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

### Why TensorFlow Lite?

TensorFlow Lite is a framework that enables the deployment of models on mobile and embedded devices. It offers several advantages:

- **Optimized for Mobile**: TFLite models are optimized to run efficiently on mobile and edge devices, using hardware acceleration when available.
- **Multi-Platform**: TFLite supports multiple platforms, including Android, iOS, and embedded Linux, making it versatile for various deployment scenarios.
- **Small Footprint**: TFLite models are smaller in size compared to their TensorFlow counterparts, helping conserve device resources.

## Inference Routine

Here are the main steps in the inference process:

1. **Model Loading**: Load the SSD MobileNetV1 model in TFLite format.
2. **Video Input**: Capture video from a camera or use a pre-recorded video file.
3. **Preprocessing**: Convert video frames to the input format required by the model.
4. **Inference**: Run the model on each frame to detect objects.
5. **Post-processing**: Decode the model's output to obtain bounding boxes and class labels.
6. **Display**: Render the detected objects on the video frames and display the video in real-time.

## Testing with Available Data

To demonstrate the model's capabilities, the inference routine was tested on videos obtained from Kaggle and on a local PC with 8 GB of RAM. The following metrics were recorded:

- **Frame Rate**: The system processed images at an average rate of 30 FPS.
- **Average Inference Time per Image**: 33 ms.
- **Memory Usage**: The memory footprint was low, with less than 2 MB of RAM used during inference.

Video reference: [Kaggle Video](https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v1/tfLite/metadata/1?lite-format=tflite&tfhub-redirect=true)

## Performance and Metrics

### Speed and Efficiency

The SSD MobileNetV1 model, when deployed with TensorFlow Lite, is very efficient in terms of speed and resource usage:

- **Latency**: Inference latency on a Pixel 4 device is about 20 ms on the GPU and 29 ms on the CPU.
- **Model Size**: The TFLite model is compact, at 3.99 MB, making it ideal for devices with limited storage space.
- **Resource Efficiency**: The model's resource efficiency ensures it can run on devices with limited power, making it suitable for real-time applications in the field.

References:
- [TensorFlow Lite Object Detection Overview](https://www.tensorflow.org/lite/examples/object_detection/overview?hl=fr)
- [TensorFlow Detection Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

## Project Structure

```bash
├── data
│   └── ...
├── evaluation
│   ├── evaluate_ssd_mobilenet_performance.ipynb
├── inference
│   ├── __init__.py
│   ├── object_detector.py
│   └── ssd_mobilenet_tflite_inference.py
├── models
│   └── ssd_mobilenet_tflite
│       ├── label_map.txt
│       └── ...
├── main.py
├── object_detection_manager.py
├── README.md
├── requirements.txt
```

- **evaluation/**
  - **evaluate_ssd_mobilenet_performance.ipynb**  
    - Notebook to evaluate the performance of the SSD MobileNetV1 model.

- **inference/**
  - **ssd_mobilenet_tflite_inference.py**  
    - **SSDMobileNetTFLiteDetector**  
      - Purpose: Executes inference using the SSD MobileNetV1 model in TFLite format. Performs predictions on an image.

  - **object_detector.py**  
    - **ObjectDetector**  
      - Purpose: Utilizes SSDMobileNetTFLiteDetector to detect objects, preprocesses the image, then draws bounding boxes and class labels on the image.
      
- **object_detection_manager.py**  
  - **ObjectDetectionManager**  
    - Purpose: Coordinates the object detection process. Manages input/output files and uses ObjectDetector to process images and video streams.
    
- **main.py**  
  - Purpose: The main entry point of the program. Configures and runs object detection based on the provided arguments (image or video stream).

## Program Usage

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Arezki93/computer_vision_challenge/
    ```

2. Navigate to the project directory:
    ```bash
    cd computer_vision_challenge
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

The `main.py` script allows for object detection on images or video streams using the SSD MobileNetV1 model in TFLite format. It accepts several arguments to configure its behavior.

#### Arguments

- `--mode` : Operation mode, either `image` for images or `stream` for videos.
- `--model` : Path to the TFLite model file.
- `--labels` : Path to the labels file.
- `--confidence` : Confidence threshold for detection.
- `--input` : Path to the input file (image or video) or camera index.
- `--output` : Path to save the processed result (image or video).

## Execution Examples

**Stream Mode**  
To process a video stream and save the result:
```bash
python main.py --mode stream --model ./models/ssd_mobilenet_tflite/ssd_mobilenet.tflite --labels ./models/label_map.txt --confidence 0.6 --input ./data/026c7465-309f6d33.mp4 --output ./data/output/026c7465-309f6d33.mp4
```




