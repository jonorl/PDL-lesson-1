from ddgs import *
from fastcore.all import *
from fastai.vision.all import *

learn = load_learner('export.pkl')

is_carrot,_,probs = learn.predict(PILImage.create('carrot.jpg'))
print(f"This is a: {is_carrot}.")
print(f"Probability it's a carrot: {probs[0]:.4f}")