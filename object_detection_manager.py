# coding=utf-8

from inference import ObjectDetector
import cv2
import time
import os
import logging

class ObjectDetectionManager:
    def __init__(self, model_path: str, labels_path: str, confidence_threshold: float = 0.5) -> None:
        """
        Initializes the ObjectDetectionManager with an ObjectDetector.

        Args:
            model_path (str): Path to the model file.
            labels_path (str): Path to the label file.
            confidence_threshold (float): Confidence threshold for detection.
        """
        try:
            self.detector = ObjectDetector(model_path, labels_path, confidence_threshold)
        except Exception as e:
            logging.error(f"Failed to initialize ObjectDetector: {e}")
            raise RuntimeError(f"Failed to initialize ObjectDetector: {e}")

    def _ensure_directory_exists(self, path: str) -> None:
        """
        Ensures that the directory for the given path exists. If not, creates it.

        Args:
            path (str): The path where the directory should be ensured.
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

    def process_image(self, image_path: str, output_path: str = None) -> None:
        """
        Processes a single image file and optionally saves the result.

        Args:
            image_path (str): Path to the image file.
            output_path (str): Path to save the processed image (optional).
        """
        try:
            image_np = cv2.imread(image_path)
            if image_np is None:
                raise FileNotFoundError(f"Image not found: {image_path}")

            result_image = self.detector.detect_and_draw(image_np)
            cv2.imshow("Detected Image", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if output_path:
                self._ensure_directory_exists(output_path)
                cv2.imwrite(output_path, result_image)
                logging.info(f"Image saved to {output_path}")

        except Exception as e:
            logging.error(f"Error processing image: {e}")

    def process_stream(self, source: str, output_path: str = None) -> None:
        """
        Processes a video file or a camera feed and optionally saves the result.

        Args:
            source (str): Path to the video file or camera index .
            output_path (str): Path to save the processed video (optional).
        """
        try:
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open camera with index {source}")
            else:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    raise FileNotFoundError(f"Failed to open video file: {source}")

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = None
            if output_path:
                self._ensure_directory_exists(output_path)
                frame_size = (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                out = cv2.VideoWriter(output_path, fourcc, 20.0, frame_size)

            prev_time = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result_frame = self.detector.detect_and_draw(frame)
                
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                
                cv2.putText(result_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Detected Stream", result_frame)

                if out:
                    out.write(result_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logging.error(f"Error processing stream: {e}")