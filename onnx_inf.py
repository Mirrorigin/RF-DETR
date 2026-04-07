"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from src.rfdetr.datasets.target_classes import TARGET_CLASSES

THRESHOLD = 0.5

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, labels, boxes, scores, ratios, paddings, thrh=THRESHOLD):
    result_images = []

    font = ImageFont.load_default()

    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb, s in zip(lab, box, scr):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline='red', width=2)

            class_name = TARGET_CLASSES[lbl]
            text = f"{class_name}: {s:.2f}"
            text_bbox = draw.textbbox((bb[0], bb[1]), text, font=font)
            draw.rectangle(text_bbox, fill='white')
            draw.text((bb[0], bb[1]), text, fill='black', font=font)
            # draw.text((bb[0], bb[1]), text=str(lbl), fill='blue')

        result_images.append(im)
    return result_images


def process_image(sess, im_pil, size=640):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, size)

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    im_data = transforms(resized_im_pil).unsqueeze(0)

    scale_fct = torch.tensor([size, size, size, size])

    output = sess.run(
        output_names=['dets', 'labels'],
        input_feed={'input': im_data.numpy()}
    )

    pred_boxes_np, pred_logits_np = output
    pred_logits = torch.from_numpy(pred_logits_np)
    probs = pred_logits.sigmoid()

    scores_tensor, labels_tensor = probs[0].max(-1)
    pred_boxes_tensor = torch.from_numpy(pred_boxes_np)[0]
    boxes_xyxy_normalized = box_cxcywh_to_xyxy(pred_boxes_tensor)
    boxes_xyxy_pixels = boxes_xyxy_normalized * scale_fct

    labels = labels_tensor.numpy()
    boxes = boxes_xyxy_pixels.numpy()
    scores = scores_tensor.numpy()

    result_images = draw(
        [im_pil], labels, boxes, scores,
        [ratio], [(pad_w, pad_h)]
    )
    result_images[0].save('onnx_result.jpg')
    print("Image processing complete. Result saved as 'result.jpg'.")


def process_video(sess, video_path, size=640):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('backyard_result.mp4', fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    prediction_results = []
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize frame while preserving aspect ratio
        resized_frame_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, size)

        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        im_data = transforms(resized_frame_pil).unsqueeze(0)

        scale_fct = torch.tensor([size, size, size, size])

        output = sess.run(
            output_names=['dets', 'labels'],
            input_feed={'input': im_data.numpy()}
        )

        # Manually processed (RF-DETR gives raw ouputs)
        pred_boxes_np, pred_logits_np = output

        pred_logits = torch.from_numpy(pred_logits_np)
        probs = pred_logits.sigmoid()
        scores_tensor, labels_tensor = probs[0].max(-1)

        pred_boxes_tensor = torch.from_numpy(pred_boxes_np)[0]
        boxes_xyxy_normalized = box_cxcywh_to_xyxy(pred_boxes_tensor)
        boxes_xyxy_pixels = boxes_xyxy_normalized * scale_fct

        labels = labels_tensor.numpy()
        boxes = boxes_xyxy_pixels.numpy()
        scores = scores_tensor.numpy()

        # Results collection
        pred_indices = scores > THRESHOLD
        pred_labels_ids = labels[pred_indices]
        pred_scores = scores[pred_indices]
        pred_names = [TARGET_CLASSES[l] for l in pred_labels_ids]

        pred_labels_str = ",".join(pred_names) if pred_names else "--"
        pred_scores_str = ",".join([f"{s:.2f}" for s in pred_scores]) if pred_names else "--"

        prediction_results.append({
            'Frame_id': frame_count+1,
            'Pred_Labels': pred_labels_str,
            'Pred_Scores': pred_scores_str
        })

        # Draw detections on the original frame
        result_images = draw(
            [frame_pil], [labels], [boxes], [scores],
            [ratio], [(pad_w, pad_h)]
        )
        frame_with_detections = result_images[0]

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'onnx_result.mp4'.")

    pred_df = pd.DataFrame(prediction_results)
    pred_df.to_csv("onnx_result.csv", index=False)
    print("Prediction results saved to 'onnx_result.csv'.")


def main(args):
    """Main function."""
    # Load the ONNX model
    sess = ort.InferenceSession(args.onnx)
    size = sess.get_inputs()[0].shape[2]
    print(f"Using device: {ort.get_device()}")

    input_path = args.input

    try:
        # Try to open the input as an image
        im_pil = Image.open(input_path).convert('RGB')
        process_image(sess, im_pil, size)
    except IOError:
        # Not an image, process as video
        process_video(sess, input_path, size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or video file.')
    args = parser.parse_args()
    main(args)
