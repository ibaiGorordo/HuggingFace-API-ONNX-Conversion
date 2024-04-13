import gradio as gr
from ultralytics import YOLOWorld

from ExportModel import ModelExporter

model_list = ['yolov8s-worldv2', 'yolov8m-worldv2', 'yolov8l-worldv2', 'yolov8x-worldv2']

def export_model(model, width, height, num_classes):
    model_name = model
    img_width = width
    img_height = height
    num_classes = num_classes

    yoloModel = YOLOWorld(model_name)
    yoloModel.set_classes([""] * num_classes)

    # Initialize model exporter
    export_model = ModelExporter(yoloModel.model)

    # Export model
    output_path = export_model.export("temp", model_name, img_width, img_height, num_classes)

    return output_path


demo = gr.Interface(
    export_model,
    [
        gr.Dropdown(model_list, label="model", value=model_list[0]),
        gr.Slider(32, 4096, step=32, value=640, label="width"),
        gr.Slider(32, 4096, step=32, value=480, label="height"),
        gr.Number(label="num_classes", value=1),
    ],
    "file",
    title="ONNX Export Ultralytics YOLO-World Open Vocabulary Object Detection",
    description="Demo to export Ultralytics YOLO-World Open Vocabulary Object Detection model to ONNX",
    api_name="export"
)
if __name__ == "__main__":
    demo.launch()
