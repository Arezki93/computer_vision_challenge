# coding=utf-8

import numpy as np
import cv2
from .ssd_mobilenet_tflite_inference import SSDMobileNetTFLiteDetector

class ObjectDetector:
    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.5) -> None:
        """
        Initializes the ObjectDetector by loading the SSD MobileNet TFLite model.

        Args:
            model_path (str): Path to the model file.
            labels_path (str): Path to the label file.
            confidence_threshold (float): Confidence threshold for detection.
            
        Raises:
            ValueError: If an unsupported model type is provided.
            RuntimeError: If the model cannot be loaded.
        """
        self.confidence_threshold = confidence_threshold
        self.labels = self.load_labels(labels_path)
        self.detector = SSDMobileNetTFLiteDetector(model_path)
        
    @staticmethod
    def load_labels(labels_path: str) -> dict:
        """
        Load labels from a file into a dictionary.

        Args:
            labels_path (str): Path to the label file.
        
        Returns:
            dict: A dictionary mapping class indices to class names.
        """
        labels = {}
        with open(labels_path, 'r') as f:
            for line in f:
                idx, label = line.strip().split(maxsplit=1)
                labels[int(idx)] = label
        return labels

    def detect_and_draw(self, image_np: np.ndarray) -> np.ndarray:
        """
        Detects objects in an image and draws bounding boxes around detected objects.

        Args:
            image_np (np.ndarray): The image to analyze as a NumPy array. Must be in RGB format.

        Returns:
            np.ndarray: The image with bounding boxes drawn around detected objects.

        Raises:
            RuntimeError: If an error occurs during inference.
        """
        try:
            input_image = cv2.resize(image_np, (300, 300))
            num_detections, boxes, classes, scores = self.detector.detect_objects(input_image)
            height, width, _ = image_np.shape
            for i in range(num_detections):
                if scores[i] > self.confidence_threshold: 
                    ymin, xmin, ymax, xmax = boxes[i]
                    xmin = int(xmin * width)
                    xmax = int(xmax * width)
                    ymin = int(ymin * height)
                    ymax = int(ymax * height)
                    
                    cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    class_id = int(classes[i])
                    label = self.labels.get(class_id, f"Classe {class_id}")
                   
                    label_with_score = f"{label}: {scores[i]:.2f}"
                    
                    cv2.putText(image_np, label_with_score, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return image_np
        except Exception as e:
            raise RuntimeError(f"Error during detection and drawing: {e}")