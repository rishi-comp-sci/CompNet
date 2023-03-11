import os
from os import path
import torch
from torchvision import transforms
from torch.nn import functional as F
# from lavis.models import load_model_and_preprocess
# from t5_utils import T5_Utils
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np

import clip

clip_model, preprocess = clip.load("ViT-B/32")
clip_model.cuda().eval()
input_resolution = clip_model.visual.input_resolution
context_length = clip_model.context_length
vocab_size = clip_model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda t: t.float())
])

clip_preproc = transforms.Compose([
    transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    transforms.CenterCrop(size=(224, 224)),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])


def getbatch(dir, paths):
    images = []
    for path in paths:
        tpath = dir + path
        images.append(Image.open(tpath).convert("RGB"))
    return images

def batchify(paths, n=100):
    batches = []
    i = 0
    while i < len(paths):
        i2 = min(i+n, len(paths))
        batches.append(paths[i:i2])
        i = i2
    return batches

@torch.no_grad()
def get_text_encodings_for_batch(images, paths, savedir):
    clip_encodings = []
    for image in images:
        with torch.no_grad():
            preproc_images = clip_preproc(torch.unsqueeze(transform(image), 0)).cuda()
            clip_encodings.append(clip_model.encode_image(preproc_images).float())
    for i in range(len(clip_encodings)):
        filename = paths[i].split('.')[0] + ".pt"
        torch.save(clip_encodings[i], savedir+filename)

if not path.exists("../../imagenet/CLIP_ENCODINGS"):
    os.mkdir("../../imagenet/CLIP_ENCODINGS")

for ds in ["train", "val", "test"]:
    savedir = "../../imagenet/CLIP_ENCODINGS/"+ds+"/"
    if not path.exists(savedir): os.mkdir(savedir)
    dir = "../../imagenet/ILSVRC/Data/CLS-LOC/"+ds+"/"
    print(ds, dir, savedir)
    if ds == "train":
        folders = os.listdir(dir)
        for fold in tqdm(folders):
            folddir = dir+fold+"/"
            paths = os.listdir(folddir)
            batches = batchify(paths)
            for batch in tqdm(batches, leave=False):
                images = getbatch(folddir, batch)
                get_text_encodings_for_batch(images, batch, savedir)
    else:
        paths = os.listdir(dir)
        batches = batchify(paths)
        for batch in batches:
            images = getbatch(dir, batch)
            get_text_encodings_for_batch(images, batch, savedir)
