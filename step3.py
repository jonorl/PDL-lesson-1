from ddgs import *
from fastcore.all import *
from fastai.vision.all import *
from fastai.imports import *

learn = load_learner('export.pkl')
print(learn.dls.vocab)

is_dog,_,probs = learn.predict(PILImage.create('cat.jpg'))
print(f"This is a: {is_dog}.")
print(f"Probability it's a dog: {probs[0]:.4f}")