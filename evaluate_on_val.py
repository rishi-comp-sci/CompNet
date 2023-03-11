import argparse
import os
from compnet_model import CompNet
from pytorch_lightning.loggers import CometLogger
import re
from compnet_data import compnet_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import json
from random import sample
import cv2
from cv2 import cvtColor, COLOR_RGB2GRAY

parser = argparse.ArgumentParser(description='Test GazeTracker')
parser.add_argument('--dataset_dir', default='../../imagenet/ILSVRC/Data/CLS-LOC/', help='Path to converted dataset')
parser.add_argument('--checkpoints', default='../../imagenet/COMPNET_CHECKPOINTS/', help='Model checkpoints')

if __name__ == '__main__':

    GRAY = lambda x: ImageOps.grayscale(x)

    def PSNR(original, compressed):
        mse = np.mean((np.asarray(original) - np.asarray(compressed)) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
                    # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    args = parser.parse_args()

    def get_val_loss(fname):
        return float(re.findall("val_loss=[-+]?(?:\d*\.\d+|\d+)", fname)[0].replace("val_loss=", ""))

    def get_best_checkpoint(chkptsdir=args.checkpoints):
        ckpts = os.listdir(chkptsdir)
        return min(ckpts, key=lambda x: get_val_loss(x))

    def euc(a, b):
        return np.sqrt(np.sum(np.square(a - b), axis=1))

    logger = CometLogger(
        api_key="zQb5zTn15DuSBvVEBjWvs2HzK",
        project_name="TEST_MODELS",
    )

    model = CompNet(None, None, 256, logger)

    if(torch.cuda.is_available()):
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')
    gbc = get_best_checkpoint()
    print("Device:", dev, "Best checkpoint:", gbc)
    weights = torch.load(args.checkpoints+gbc, map_location=dev)['state_dict']
    model.load_state_dict(weights)
    model.to(dev)
    model.eval()

    for ds in ['val']:
        if not os.path.exists(f"ORIGINAL_IMAGES_EXAMPLES/gray_{ds}/"): os.mkdir(f"ORIGINAL_IMAGES_EXAMPLES/gray_{ds}/")
        if not os.path.exists(f"JPEG_IMAGES_EXAMPLES/gray_{ds}/"): os.mkdir(f"JPEG_IMAGES_EXAMPLES/gray_{ds}/")
        if not os.path.exists(f"RECON_IMAGES_EXAMPLES/gray_{ds}/"): os.mkdir(f"RECON_IMAGES_EXAMPLES/gray_{ds}/")
        # input(ds)
        file_root = args.dataset_dir+ds+"/"
        dataset = compnet_dataset(phase=ds)
        dataloader = DataLoader(dataset, batch_size=16, num_workers=8, pin_memory=False, shuffle=False)
        results = {'compnet_psnr':[], 'compnet_ssim':[], 'jpeg_psnr':[], 'jpeg_ssim':[]}

        save_idxs = sample([i for i in range(len(dataset))], 50)
        curr_idx = 0

        for j in tqdm(dataloader):
            with torch.no_grad():
                image, clip_enc = j[0].to(dev), j[2].to(dev)
                t = torch.zeros(16,).to(dev)
                x = model.forward_diffusion_sample(image, t).to(dev)
                x = torch.dequantize(torch.quantize_per_tensor(torch.nn.functional.interpolate(torch.nn.functional.interpolate(x, size=64, mode='bicubic'), size=256, mode='bicubic'), 0.1, 10, torch.qint8)).to(dev)
                y = model.get_previous(image, t).to(dev)
                y_hat = model(x, t, clip_enc).to(dev)
                outs = (x-y_hat).cpu().detach().numpy()

                image = image.cpu().detach().numpy()

                for i in tqdm(range(image.shape[0]), leave=False):
                    get_img = lambda x: np.moveaxis(x, 0, -1)
                    IMG = (get_img((image[i]+1.0)/2.0)*255.0).astype(np.uint8)
                    IMG = GRAY(Image.fromarray(IMG))
                    
                    get_rgb_mins = lambda x: np.expand_dims(np.expand_dims(np.min(x, axis=(-2, -1)), axis=-1), axis=-1)
                    get_rgb_maxes = lambda x: np.expand_dims(np.expand_dims(np.max(x, axis=(-2, -1)), axis=-1), axis=-1)
                    out = outs[i]
                    out = out - get_rgb_mins(out)
                    out = out / get_rgb_maxes(out)
                    OUT = GRAY(Image.fromarray((get_img(out)*255.0).astype(np.uint8)))
                    compnet_psnr = PSNR(IMG, OUT)
                    compnet_ssim = ssim(np.asarray(IMG), np.asarray(OUT))

                    IMG.save('gray_compressed_2.jpeg', quality=1)
                    img_jpeg = GRAY(Image.open('gray_compressed_2.jpeg'))
                    jpeg_psnr = PSNR(IMG, img_jpeg)
                    jpeg_ssim = ssim(np.asarray(IMG), np.asarray(img_jpeg))

                    results['compnet_psnr'].append(compnet_psnr)
                    results['compnet_ssim'].append(compnet_ssim)
                    results['jpeg_psnr'].append(jpeg_psnr)
                    results['jpeg_ssim'].append(jpeg_ssim)

                    if curr_idx in save_idxs:
                        IMG.save(f"ORIGINAL_IMAGES_EXAMPLES/gray_{ds}/{curr_idx}.png")
                        OUT.save(f"RECON_IMAGES_EXAMPLES/gray_{ds}/{curr_idx}.png")
                        img_jpeg.save(f"JPEG_IMAGES_EXAMPLES/gray_{ds}/{curr_idx}.jpeg")

                    curr_idx += 1

        mean_compnet_psnr = np.mean(results['compnet_psnr'])
        mean_compnet_ssim = np.mean(results['compnet_ssim'])
        mean_jpeg_psnr = np.mean(results['jpeg_psnr'])
        mean_jpeg_ssim = np.mean(results['jpeg_ssim'])

        median_compnet_psnr = np.median(results['compnet_psnr'])
        median_compnet_ssim = np.median(results['compnet_ssim'])
        median_jpeg_psnr = np.median(results['jpeg_psnr'])
        median_jpeg_ssim = np.median(results['jpeg_ssim'])

        print(ds + " results (gray):")

        print("\tCOMPNET:")
        print(f"\t\tMEAN PSNR: {mean_compnet_psnr}\tMEAN SSIM: {mean_compnet_ssim}")
        print(f"\t\tMEDIAN PSNR: {median_compnet_psnr}\MEDIAN SSIM: {median_compnet_ssim}")

        print("\tJPEG 2000:")
        print(f"\t\tMEAN PSNR: {mean_jpeg_psnr}\tMEAN SSIM: {mean_jpeg_ssim}")
        print(f"\t\tMEDIAN PSNR: {median_jpeg_psnr}\MEDIAN SSIM: {median_jpeg_ssim}")

        json_path = ds+"_results_gray.json"
        with open(json_path, 'w') as outfile:
            json.dump(results, outfile)
