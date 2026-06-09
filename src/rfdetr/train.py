from rfdetr import RFDETRBase
from rfdetr.datasets.target_classes import TARGET_CLASSES

model = RFDETRBase()

model.train(
    dataset_dir='/Shared/nas4321/projects/intellisee/Master_1.71',
    dataset_file='intellisee',
    target_classes=TARGET_CLASSES,
    epochs=50,
    batch_size=4,
    grad_accum_steps=2, # maintain a total batch size of 16
    lr=1e-4,
    output_dir="./output/",
)