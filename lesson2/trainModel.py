from ddgs import *
from fastcore.all import *
from fastai.vision.all import *

## Step 2: Train our model

path = Path('fugazzeta_or_not_fugazzeta')

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

print("Creating dataloaders")
pizza = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
# Perform Data augumentation
pizza = pizza.new(item_tfms=Resize(128), batch_tfms=aug_transforms())

dls = pizza.dataloaders(path, bs=32)
#dls = dls.train.show_batch(max_n=4, nrows=1, unique=True)
print("Dataloaders created")

print("Starting training")
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
print("Training complete")
print("Exporting model")
learn.export()

print(f"Model is on: {next(learn.model.parameters()).device}")