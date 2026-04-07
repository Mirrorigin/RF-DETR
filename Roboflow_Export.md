# Get onnx file

Download model via API:

```python
from inference.models.utils import get_model

model = get_model(model_id="intellisee-master-dataset/156", api_key="***")

```

Then go to `/tmp/cache/models-cache/intellisee-master-dataset-*`, where you can find `weights.onnx` file.

# Check onnx file

```shell
pip install netron
netron weights.onnx
```
Then we can check model architecture at http://localhost:8080.

Onnx run example:
```python
import onnxruntime as ort

session = ort.InferenceSession("weights.onnx")
input_name = session.get_inputs()[0].name   # input
input_shape = session.get_inputs()[0].shape # [1, 3, 704, 704]

def preprocess(img_path, target_size=(640, 640)):
    ...

input_tensor, original_shape = preprocess("test_image.jpg")
outputs = session.run(None, {input_name: input_tensor})
```

# Export as TensorRT

```shell
trtexec --onnx=weights.onnx --saveEngine=rfdetr.engine --fp16
```

# Run Inference

Check `trt_inference.py` File.