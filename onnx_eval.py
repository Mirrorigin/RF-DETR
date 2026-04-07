import numpy as np
import os
import cv2
import onnxruntime as ort
from glob import glob
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLS_NAMES = ["__background__", "backpack", "broom", "cellphone", "fire", "handgun", "longgun",
             "person_fallen", "person_sitting", "person_standing", "reflection", "smoke", "snow",
             "spill", "vehicle"]

ONNX_WEIGHTS = "/home/jingmliang/Projects/RF-DETR/src/rfdetr/best_models/Roboflow_RFDETR_weights.onnx"
TEST_DIR = "/home/jingmliang/Downloads/Intellisee master dataset.v156-rf-detr-v1.74.coco/test"
# TEST_DIR = "/Shared/nas4321/projects/intellisee/Master_1.71/test"
ANN_PATH = os.path.join(TEST_DIR, "_annotations.coco.json")
BATCH_SIZE = 1
CONF_THRESHOLD = 0.001
INPUT_SIZE = (704, 704)

def xyxy_to_xywh(box_xyxy):
    x1, y1, x2, y2 = map(float, box_xyxy)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def remapped_coco(coco_ann_path, target_classes):
    """Remap COCO annotations to target_classes (id=0...N-1)，and return new COCO"""
    coco = COCO(coco_ann_path)
    name_to_new_id = {name: i for i, name in enumerate(target_classes)}

    orig_id_to_name = {cid: cat['name'] for cid, cat in coco.cats.items()}
    new_anns = []
    kept_image_ids = set()

    for ann in coco.anns.values():
        orig_cid = ann['category_id']
        name = orig_id_to_name.get(orig_cid)
        if name in name_to_new_id:
            remap_ann = ann.copy()
            remap_ann['category_id'] = name_to_new_id[name]
            new_anns.append(remap_ann)
            kept_image_ids.add(ann['image_id'])

    info = coco.dataset.get('info', {"description": "remapped for eval"})
    licenses = coco.dataset.get('licenses', [])

    new_images = [coco.imgs[i] for i in sorted(kept_image_ids)]
    new_cats = [{'id': i, 'name': name} for i, name in enumerate(target_classes)]

    remap = COCO()
    remap.dataset = {
        'info': info,
        'licenses': licenses,
        'images': new_images,
        'annotations': new_anns,
        'categories': new_cats,
    }
    remap.createIndex()
    return remap

class ONNXPredictor:
    def __init__(self, onnx_path, input_size=(704, 704)):
        # Initialize the ONNX Runtime session
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

        # ImageNet mean and std for normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, image_paths):
        batch_images = []
        orig_sizes = []

        for p in image_paths:
            # Read image (OpenCV reads in BGR format by default)
            img = cv2.imread(p)
            if img is None:
                continue

            # Convert BGR to RGB (PyTorch's Image.open uses RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Record original height and width (H, W) for bounding box post-processing
            orig_sizes.append(img.shape[:2])

            # Resize to the fixed input size of the model (e.g., 704x704)
            img = cv2.resize(img, self.input_size)

            # Pixel value normalization
            # Scale integer pixel values (0~255) to float values (0.0~1.0)
            img = img.astype(np.float32) / 255.0
            # Subtract mean and divide by standard deviation (ImageNet standards). This is also done in RFDETR.predit.
            img = (img - self.mean) / self.std

            # Dimension conversion: HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
            # ONNX expects input shape: [Batch, Channel, Height, Width]
            img = np.transpose(img, (2, 0, 1))
            batch_images.append(img)

        # 6. Stack the list of individual images into a single batch, shape: (B, 3, H, W)
        return np.stack(batch_images), orig_sizes

    def predict(self, batch_paths, threshold=0.001):
        batch_data, orig_sizes = self.preprocess(batch_paths)

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: batch_data})

        # outputs[0] --> boxes and outputs[1] --> logits.
        pred_boxes = outputs[0]  # Expected shape: (B, N, 4)
        pred_logits = outputs[1]  # Expected shape: (B, N, num_classes)

        # Convert logits to probabilities using Sigmoid
        pred_scores = 1 / (1 + np.exp(-pred_logits))

        dets_list = []
        for b in range(len(batch_paths)):
            scores_b = pred_scores[b]
            boxes_b = pred_boxes[b]

            # Get the highest score and corresponding class ID for each query
            class_ids = np.argmax(scores_b, axis=-1)
            confidences = np.max(scores_b, axis=-1)

            # Filter out low-confidence predictions
            keep_idx = confidences > threshold

            keep_boxes = boxes_b[keep_idx]
            keep_class_ids = class_ids[keep_idx]
            keep_confs = confidences[keep_idx]

            orig_h, orig_w = orig_sizes[b]

            scaled_boxes = []
            for box in keep_boxes:
                # DETR outputs normalized [cx, cy, w, h] in the range [0, 1]
                cx, cy, bw, bh = box[0], box[1], box[2], box[3]

                # Convert normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
                # mapped directly to the original image dimensions
                x1 = (cx - bw / 2.0) * orig_w
                y1 = (cy - bh / 2.0) * orig_h
                x2 = (cx + bw / 2.0) * orig_w
                y2 = (cy + bh / 2.0) * orig_h

                scaled_boxes.append([x1, y1, x2, y2])

            scaled_boxes = np.array(scaled_boxes) if len(scaled_boxes) > 0 else np.empty((0, 4))

            # Store results in a dictionary format
            dets_list.append({
                "xyxy": scaled_boxes,
                "class_id": keep_class_ids,
                "confidence": keep_confs
            })

        return dets_list

print(f"Total {len(CLS_NAMES)} classes!")
coco_gt = remapped_coco(ANN_PATH, target_classes=CLS_NAMES)
fname_to_imgid = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
name_to_catid = {cat["name"]: cat["id"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

total_found = sorted([p for p in glob(os.path.join(TEST_DIR, "*")) if
                      os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
image_paths = [p for p in total_found if os.path.basename(p) in fname_to_imgid]
print(f"Eval images after filter: {len(image_paths)} / original {len(total_found)}")

# Initialize ONNX Model
model = ONNXPredictor(onnx_path=ONNX_WEIGHTS, input_size=INPUT_SIZE)

coco_results = []
for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Predicting (ONNX)"):
    batch_paths = image_paths[i:i + BATCH_SIZE]

    # Predict
    dets_list = model.predict(batch_paths, threshold=CONF_THRESHOLD)

    for p, det in zip(batch_paths, dets_list):
        fname = os.path.basename(p)
        if fname not in fname_to_imgid:
            continue

        image_id = fname_to_imgid[fname]

        for (x1, y1, x2, y2), cid, score in zip(det["xyxy"], det["class_id"], det["confidence"]):
            cid = int(cid)

            if 0 <= cid < len(CLS_NAMES):
                cls_name = CLS_NAMES[cid]
                if cls_name in name_to_catid:
                    category_id = int(name_to_catid[cls_name])
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": xyxy_to_xywh([x1, y1, x2, y2]),
                        "score": float(score),
                    })

if not coco_results:
    print("Warning: No detections found above threshold!")
else:
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50 = coco_eval.stats[1]
    print(f"mAP @[ IoU=0.50 ]:  {map_50:.4f}")

    prec = coco_eval.eval['precision']
    t = np.where(np.isclose(coco_eval.params.iouThrs, 0.50))[0][0]
    a = coco_eval.params.areaRngLbl.index('all')
    m = coco_eval.params.maxDets.index(coco_eval.params.maxDets[-1])

    per_class_ap50 = []
    for k in range(len(coco_eval.params.catIds)):
        p = prec[t, :, k, a, m]
        p = p[p > -1]
        ap50 = float(np.mean(p)) if p.size else float('nan')
        per_class_ap50.append(ap50)

    cats = coco_gt.loadCats(coco_eval.params.catIds)
    names = [c['name'] for c in cats]
    ap50_by_class = dict(zip(names, per_class_ap50))

    print("-" * 40)
    for name, ap in sorted(ap50_by_class.items()):
        if not np.isnan(ap):
            print(f"{name:30s} {ap:.4f}")