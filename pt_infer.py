import cv2
import torch
import numpy as np


BEST_WEIGHTS = "/home/jingmliang/Projects/RF-DETR/src/rfdetr/best_models/checkpoint_bigvision_latest_v1.pt"

IMAGE_IN = "/home/jingmliang/Temp/DEBUG_IMG/gun/image_3.jpg"
IMAGE_OUT = "rfdetr_image_output.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 704
CONF_THRESH = 0.1

CLASS_NAMES = [
    "__background__",
    "backpack",
    "broom",
    "cellphone",
    "handgun",
    "longgun",
    "person_fallen",
    "person_sitting",
    "person_standing",
    "reflection",
    "snow",
    "spill",
    "vehicle",
]

def preprocess_bgr(frame_bgr, input_size=704, use_imagenet_norm=False):
    orig_h, orig_w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    img = resized.astype(np.float32) / 255.0

    # use_imagenet_norm=True: Assume that the Normalize preprocessing already be included inside the exported model.
    if use_imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    images = torch.from_numpy(img).unsqueeze(0)

    orig_target_sizes = torch.tensor([[orig_h, orig_w]], dtype=torch.int64)

    return images, orig_target_sizes


def draw_detections(image_bgr, labels, boxes, scores, conf_thresh=0.5):
    h, w = image_bgr.shape[:2]

    labels = labels.detach().cpu().reshape(-1)
    boxes = boxes.detach().cpu().reshape(-1, 4)
    scores = scores.detach().cpu().reshape(-1)

    kept = 0

    # BGR colors for OpenCV
    box_color = (0, 255, 255)  # yellow/cyan-ish bright color in BGR
    text_color = (255, 255, 255)  # white
    bg_color = (0, 0, 0)  # black background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    box_thickness = 2

    for label, box, score in zip(labels, boxes, scores):
        score_val = float(score)
        if score_val < conf_thresh:
            continue

        cls_id = int(label)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        x1, y1, x2, y2 = box.tolist()

        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        text = f"{cls_name} {score_val:.2f}"

        # Draw bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), box_color, box_thickness)

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            font_thickness,
        )

        # Put label above box if possible, otherwise inside
        text_x = x1
        text_y = y1 - 8

        if text_y - text_h - baseline < 0:
            text_y = y1 + text_h + baseline + 8

        # Background rectangle coordinates
        bg_x1 = text_x
        bg_y1 = text_y - text_h - baseline
        bg_x2 = min(w - 1, text_x + text_w + 8)
        bg_y2 = min(h - 1, text_y + baseline + 4)

        cv2.rectangle(
            image_bgr,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            bg_color,
            thickness=-1,
        )

        # Draw text
        cv2.putText(
            image_bgr,
            text,
            (text_x + 4, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        kept += 1

    return image_bgr, kept


def main():
    print("Loading model...")
    model = torch.jit.load(BEST_WEIGHTS, map_location=DEVICE)
    model.eval()

    print("Model loaded.")
    print("Device:", DEVICE)
    print("Forward schema:", model.forward.schema)

    image = cv2.imread(IMAGE_IN)
    if image is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_IN}")

    images, orig_target_sizes = preprocess_bgr(
        image,
        input_size=INPUT_SIZE,
        use_imagenet_norm=False,
    )

    images = images.to(DEVICE)
    orig_target_sizes = orig_target_sizes.to(DEVICE)

    with torch.inference_mode():
        labels, boxes, scores = model(images, orig_target_sizes)

    drawn, kept = draw_detections(
        image,
        labels,
        boxes,
        scores,
        conf_thresh=CONF_THRESH,
    )

    cv2.imwrite(IMAGE_OUT, drawn)
    print(f"Kept detections above {CONF_THRESH}: {kept}")
    print(f"Saved result to: {IMAGE_OUT}")


if __name__ == "__main__":
    main()