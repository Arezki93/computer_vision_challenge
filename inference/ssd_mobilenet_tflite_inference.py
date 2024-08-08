# coding=utf-8

import numpy as np
import tensorflow as tf
from typing import Tuple

class SSDMobileNetTFLiteDetector:
    def __init__(self, model_path: str):
        """
        Initializes the SSD MobileNet tflite detector by loading the model from the specified path.

        Args:
            model_path (str): Path to the TFLite model file.

        Raises:
            RuntimeError: If an error occurs while loading the model.
        """
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            raise RuntimeError(f"Error loading the model: {e}")

    def detect_objects(self, image_np: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detects objects in an RGB image using the SSD MobileNet tflite model.

        Args:
            image_np (np.ndarray): The image to analyze as a NumPy array. The image must be in RGB format 
                                   with a shape of (300, 300, 3).

        Raises:
            ValueError: If the provided image is invalid or does not meet the required conditions.
            RuntimeError: If an error occurs during inference.

        Returns:
            Tuple[int, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the number of detections,
            detection boxes, detected classes, and detection scores.
        """
        
        try:
            if image_np is None:
                raise ValueError("The provided image is invalid or was not loaded correctly.")
        
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                raise ValueError("The image must be in RGB format with 3 channels.")
    
            if image_np.shape[0] != 300 or image_np.shape[1] != 300:
                raise ValueError("The image must be 300x300 pixels in size.")

            input_tensor = np.expand_dims(image_np, axis=0).astype(np.uint8)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()

            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            num_detections = int(self.interpreter.get_tensor(self.output_details[3]['index'])[0])

            return num_detections, boxes, classes, scores
            
        except Exception as e:
            raise RuntimeError(f"Error during inference with SSD SSDMobileNetTFLiteDetector: {e}")