# HuggingFace-API-ONNX-Conversion
 Repository to showcase how to export models to ONNX using a client to a Hugging Face space.
![huggingface-onnx](https://github.com/ibaiGorordo/HuggingFace-API-ONNX-Conversion/assets/43162939/573d11c4-84e6-4b7a-97bc-cba4ec3ba2b8)


# Why
 - Often times, to share ONNX inference models, it is necessary to install Pytorch and ONNX for the model to be converted unless you use something like a Google Colab notebook.
 - This repository showcases how to convert models to ONNX using a client to a Hugging Face space containing the necessary code to convert the model to ONNX so that only the **gradio-client** library is necessary to convert the model.

# How it works
 - Hugging Face provides a simple way to host a space with the necessary code to convert models to ONNX (apart from other functionalities like inference).
 - Additionally, Gradio provides a simple way to create a client to interact with the space.
 - This repository showcases how to create a client to interact with the space (https://huggingface.co/spaces/ibaiGorordo/Ultralytics-YOLO-World-ONNX-Export) and convert the Ultralytics YOLO-World model to ONNX.

# Requirements
Check the requirements.txt file for more information.
 - gradio-client

# Installation
```bash
git clone https://github.com/ibaiGorordo/HuggingFace-API-ONNX-Conversion.git
cd HuggingFace-API-ONNX-Conversion
pip install -r requirements.txt
```

# Usage
To convert the model to ONNX, run the following command:
```bash
python huggingface-onnx-export.py
```
**Parameters:**
 - **img_width**: Width of the input image.
 - **img_height**: Height of the input image.
 - **num_classes**: Number of classes in the model.
 - **model_name**: Name of the model to convert to ONNX.
 - **output_dir**: Directory to save the ONNX model.

# Gradio app example
You can find the Gradio app example in the **gradio-app** folder. To run the Gradio app, run the following command:
```bash
pip install -r gradio-app/requirements.txt
python gradio-app/app.py
```

# References
 - **Example Huggingface space:** https://huggingface.co/spaces/ibaiGorordo/Ultralytics-YOLO-World-ONNX-Export
 - **Gradio API client:** https://www.gradio.app/guides/getting-started-with-the-python-client
