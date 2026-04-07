import numpy as np
import os
from glob import glob
from typing import List, Dict
from tqdm import tqdm

from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BEST_WEIGHTS = "/home/jingmliang/Projects/RF-DETR/src/rfdetr/best_models/checkpoint_bigvision.pth"
TEST_DIR     = "/home/jingmliang/Downloads/Intellisee master dataset.v156-rf-detr-v1.74.coco/test"
# TEST_DIR     = "/Shared/nas4321/projects/intellisee/Master_1.71/test"
ANN_PATH     = os.path.join(TEST_DIR, "_annotations.coco.json")
BATCH_SIZE   = 8
CONF_THRESHOLD = 0.001

# Utils func

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

    # Only keep images with targets
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

# Load model
model = RFDETRBase(
    pretrain_weights=BEST_WEIGHTS,
    patch_size = 16,
    num_windows = 2,
    dec_layers = 4,
    out_feature_indexes = [3, 6, 9, 12],
    resolution = 704,
    positional_encoding_size = 44
)

# Get class names
if hasattr(model.model, "class_names") and model.model.class_names:
    cls_names: List[str] = list(model.model.class_names)
else:
    cls_names = [name for _, name in sorted(COCO_CLASSES.items())]

print(f"Total {len(cls_names)} classes!")

coco_gt = remapped_coco(ANN_PATH, target_classes=cls_names)
fname_to_imgid = {img["file_name"]: img["id"] for img in coco_gt.dataset["images"]}
name_to_catid  = {cat["name"]: cat["id"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

total_found = sorted([p for p in glob(os.path.join(TEST_DIR, "*")) if os.path.splitext(p)[1].lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]])
image_paths = [p for p in total_found if os.path.basename(p) in fname_to_imgid]
print(f"Eval images after filter: {len(image_paths)} / original {len(total_found)}")

coco_results = []
for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Predicting"):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    dets_list = model.predict(batch_paths, threshold=CONF_THRESHOLD)
    if not isinstance(dets_list, list):
        dets_list = [dets_list]

    for p, det in zip(batch_paths, dets_list):
        fname = os.path.basename(p)
        if fname not in fname_to_imgid:
            continue

        image_id = fname_to_imgid[fname]

        # det.xyxy: (N,4), det.class_id: (N,), det.confidence: (N,)
        for (x1,y1,x2,y2), cid, score in zip(det.xyxy, det.class_id, det.confidence):
            cid = int(cid) - 1 # Bigvision team: 0 - background
            if 0 <= cid < len(cls_names):
                cls_name = cls_names[cid]
                if cls_name in name_to_catid:
                    category_id = int(name_to_catid[cls_name])
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": xyxy_to_xywh([x1, y1, x2, y2]),
                        "score": float(score),
                    })


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

for name, ap in sorted(ap50_by_class.items()):
    if not np.isnan(ap):
        print(f"{name:30s} {ap:.4f}")