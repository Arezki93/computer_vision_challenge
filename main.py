import argparse
import logging
from object_detection_manager import ObjectDetectionManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Object Detection with SSD MobileNet")
    parser.add_argument('--mode', choices=['image', 'stream'], required=True, help="Mode of operation: 'image' or'stream'")
    parser.add_argument('--model', required=True, help="Path to the TFLite model file")
    parser.add_argument('--labels', required=True, help="Path to the labels file")
    parser.add_argument('--confidence', type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument('--input', help="Path to the input image or video file, or camera index")
    parser.add_argument('--output', help="Path to save the processed output (image or video)")

    args = parser.parse_args()

    try:
        manager = ObjectDetectionManager(args.model, args.labels, args.confidence)
    except RuntimeError as e:
        logging.error(f"Failed to initialize ObjectDetectionManager: {e}")
        return

    if args.mode == "image":
        if not args.input:
            logging.error("Input image path must be specified for image mode.")
            return
        if args.output:
            manager.process_image(args.input, args.output)
        else:
            manager.process_image(args.input)

    elif args.mode == "stream":
        if not args.input:
            logging.error("Input video path must be specified for video mode.")
            return
        if args.output:
            manager.process_stream(args.input, args.output)
        else:
            manager.process_stream(args.input)

if __name__ == "__main__":
    main()
