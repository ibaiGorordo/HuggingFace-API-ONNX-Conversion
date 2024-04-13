import torch
import os
from copy import deepcopy

class ModelExporter(torch.nn.Module):
    def __init__(self, yoloModel, device='cpu'):
        super(ModelExporter, self).__init__()
        model = deepcopy(yoloModel).to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        self.model = model
        self.device = device

    def forward(self, x, txt_feats):
        return self.model.predict(x, txt_feats=txt_feats)

    def export(self, output_dir, model_name, img_width, img_height, num_classes):
        x = torch.randn(1, 3, img_width, img_height, requires_grad=False).to(self.device)
        txt_feats = torch.randn(1, num_classes, 512, requires_grad=False).to(self.device)

        print(x.shape, txt_feats.shape)

        # Export model
        onnx_name = model_name + ".onnx"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{onnx_name}"
        with torch.no_grad():
            torch.onnx.export(self,
                              (x, txt_feats),
                              output_path,
                              do_constant_folding=True,
                              opset_version=17,
                              input_names=["images", "txt_feats"],
                              output_names=["output"])

        return output_path