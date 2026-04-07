import numpy as np
import cv2
import os
import torch
import tensorrt as trt

CLASS_NAMES = {
    1: "backpack",
    2: "broom",
    3: "cellphone",
    4: "fire",
    5: "handgun",
    6: "longgun",
    7: "person_fallen",
    8: "person_sitting",
    9: "person_standing",
    10: "reflection",
    11: "smoke",
    12: "snow",
    13: "spill",
    14: "vehicle"
}

# load Engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)

with open("/home/jingmliang/Projects/RF-DETR/src/rfdetr/best_models/rfdetr.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_shape = (1, 3, 704, 704)
dets_shape = (1, 300, 4)
labels_shape = (1, 300, 16)

# Fetch tensor names from the engine
input_name = engine.get_tensor_name(0)
dets_name = engine.get_tensor_name(1)
labels_name = engine.get_tensor_name(2)

# Explicitly set input shape (Required for dynamic shapes)
context.set_input_shape(input_name, input_shape)

# Allocate contiguous VRAM using PyTorch
d_input = torch.empty(input_shape, dtype=torch.float32, device='cuda')
d_dets = torch.empty(dets_shape, dtype=torch.float32, device='cuda')
d_labels = torch.empty(labels_shape, dtype=torch.float32, device='cuda')

# Bind VRAM addresses to the execution context
context.set_tensor_address(input_name, d_input.data_ptr())
context.set_tensor_address(dets_name, d_dets.data_ptr())
context.set_tensor_address(labels_name, d_labels.data_ptr())

# Create the custom CUDA stream globally to manage GPU tasks
custom_stream = torch.cuda.Stream()

def preprocess_frame(frame, target_size=(704, 704)):
    orig_h, orig_w = frame.shape[:2]

    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img_rgb, target_size)

    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0

    img_chw = np.transpose(img_normalized, (2, 0, 1))   # [H, W, C] -> [C, H, W]
    img_batch = np.expand_dims(img_chw, axis=0) # [C, H, W] -> [1, C, H, W]

    img_contiguous = np.ascontiguousarray(img_batch)

    return img_contiguous, frame, orig_w, orig_h

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess_and_draw(orig_img, dets, labels, orig_w, orig_h, conf_threshold=0.5):
    # (1, 300, X) --> (300, X)
    boxes = dets[0]
    scores = sigmoid(labels[0])

    for i in range(len(boxes)):
        # Find Highest score across all classes
        class_scores = scores[i]
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]

        # Filter low-confidence predictions
        if confidence > conf_threshold:
            cx, cy, w, h = boxes[i]

            # Remap normalized (0~1) coordinates back to original absolute coordinates
            x1 = int((cx - w / 2) * orig_w)
            y1 = int((cy - h / 2) * orig_h)
            x2 = int((cx + w / 2) * orig_w)
            y2 = int((cy + h / 2) * orig_h)

            # Draw bbox
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            label_text = f"{class_name} {confidence:.2f}"

            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            label_y_pos = max(y1, text_h + 5)
            cv2.rectangle(orig_img, (x1, label_y_pos - text_h - 5), (x1 + text_w, label_y_pos + baseline - 5), (0, 255, 0), -1)

            cv2.putText(orig_img, label_text, (x1, label_y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            print(f"Object detected -> class: {class_id}, confidence: {confidence:.2f}, location: [{x1}, {y1}, {x2}, {y2}]")

    return orig_img

def process_image(input_img_path, output_img_path, conf_threshold=0.5):
    print(f"Processing Image: {input_img_path}")
    frame = cv2.imread(input_img_path)
    if frame is None:
        raise ValueError(f"Failed to open image: {input_img_path}")

    input_tensor, orig_img, orig_w, orig_h = preprocess_frame(frame)
    out_dets, out_labels = infer(input_tensor)
    result_frame = postprocess_and_draw(orig_img, out_dets, out_labels, orig_w, orig_h, conf_threshold)

    cv2.imwrite(output_img_path, result_frame)
    print(f"Image processing complete! Saved to {output_img_path}")

def process_video(input_video_path, output_video_path, conf_threshold=0.5):
    print(f"Opening video: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    print(f"Total frames to process: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process current frame
        input_tensor, orig_img, orig_w, orig_h = preprocess_frame(frame)

        # Inference execution
        out_dets, out_labels = infer(input_tensor)

        # Post-process
        result_frame = postprocess_and_draw(orig_img, out_dets, out_labels, orig_w, orig_h, conf_threshold)

        # Write output video
        out.write(result_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete! Saved to {output_video_path}")

# Run inference
def infer(input_image_data):
    d_input.copy_(torch.from_numpy(input_image_data))

    # Asynchronous execution: automatically reads/writes using the addresses bound earlier.
    context.execute_async_v3(stream_handle=custom_stream.cuda_stream)

    # Synchronization: forces the CPU to wait until the GPU has fully finished computing the outputs.
    custom_stream.synchronize()

    return d_dets.cpu().numpy(), d_labels.cpu().numpy()

if __name__ == "__main__":
    input_media_path = "/home/jingmliang/Projects/Assets/Backyard_Home_1_with_frame_id.mp4"

    # test.mp4 -> test_trt_result.mp4
    base_name, ext = os.path.splitext(os.path.basename(input_media_path))
    output_media_path = f"{base_name}_trt_result{ext}"

    if not os.path.exists(input_media_path):
        raise FileNotFoundError(f"Input file not found: {input_media_path}")

    ext = os.path.splitext(input_media_path)[-1].lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    if ext in image_extensions:
        process_image(input_media_path, output_media_path, conf_threshold=0.5)
    elif ext in video_extensions:
        process_video(input_media_path, output_media_path, conf_threshold=0.5)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Please provide a valid image or video.")