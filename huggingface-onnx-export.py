from argparse import ArgumentParser
from gradio_client import Client
import shutil
import os

def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--img_width", type=int, default=640)
    parser.add_argument("--img_height", type=int, default=480)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="yolov8l-worldv2",
                        choices=["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2"])
    parser.add_argument("--output_dir", type=str, default="models")


    args = parser.parse_args()
    img_width = args.img_width
    img_height = args.img_height
    num_classes = args.num_classes
    model_name = args.model_name
    output_dir = args.output_dir

    client = Client("ibaiGorordo/Ultralytics-YOLO-World-ONNX-Export")

    print("Exporting model... (it will take a while)")
    result = client.predict(
        model=model_name,
        width=img_width,
        height=img_height,
        num_classes=num_classes,
        api_name="/export"
    )

    os.makedirs(output_dir, exist_ok=True)
    shutil.move(result, f"{output_dir}/{model_name}.onnx")
    print(f"Model exported to {output_dir}/{model_name}.onnx")


if __name__ == "__main__":
    main()