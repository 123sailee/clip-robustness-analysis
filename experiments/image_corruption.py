"""
Image corruption experiments: apply controlled corruptions (Gaussian noise,
motion blur, JPEG compression, brightness/contrast) and evaluate CLIP zero-shot.
"""
import argparse
import os
import io
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

from torchvision import datasets

from utils.model import load_clip, pil_to_clip
from utils.metrics import top1_accuracy
from utils.visualization import plot_degradation
from utils.visualization import show_failure_cases


def add_gaussian_noise(img: Image.Image, severity: int):
    arr = np.array(img).astype(np.float32)
    sigma = severity * 10.0
    noise = np.random.randn(*arr.shape) * sigma
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def motion_blur(img: Image.Image, severity: int):
    # build a simple linear kernel
    size = 3 + severity*2
    kernel = np.zeros((size, size))
    kernel[size//2, :] = 1.0/size
    pil_kernel = ImageFilter.Kernel((size,size), kernel.flatten(), scale=1)
    return img.filter(pil_kernel)


def jpeg_compress(img: Image.Image, severity: int):
    q = max(5, 95 - severity*18)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=int(q))
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def adjust_brightness_contrast(img: Image.Image, severity: int):
    enh_b = ImageEnhance.Brightness(img)
    enh_c = ImageEnhance.Contrast(img)
    b = 1.0 - 0.15*severity
    c = 1.0 - 0.15*severity
    return enh_c.enhance(c).convert('RGB') if False else enh_b.enhance(b)


def load_dataset(root):
    return datasets.CIFAR10(root=root, train=False, download=True)


def evaluate_corruptions(dataset, model, preprocess, device, results_dir):
    classnames = dataset.classes
    # compute per-image features once
    images = [img for img,_ in dataset]
    labels = [lbl for _,lbl in dataset]

    # Use three severity levels: 1=light,2=medium,3=strong
    severity_levels = [1,2,3]
    corruptions = {
        'gaussian': add_gaussian_noise,
        'jpeg': jpeg_compress,
    }

    results = {}
    import clip
    # precompute text features for CIFAR classes
    text_prompts = [f"a photo of a {c}." for c in classnames]
    tokenized = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokenized).float()

    for name, fn in corruptions.items():
        accuracies = []
        for sev in severity_levels:
            imgs_cor = [fn(img, sev) for img in images]

            # encode in batches
            batch_size = 64
            all_preds = []
            all_logits = []
            for i in range(0, len(imgs_cor), batch_size):
                batch = imgs_cor[i:i+batch_size]
                imgs_t = torch.cat([preprocess(im).unsqueeze(0) for im in batch]).to(device)
                with torch.no_grad():
                    img_feats = model.encode_image(imgs_t).float()
                img_norm = img_feats / img_feats.norm(dim=1, keepdim=True)
                logits = img_norm @ text_feats.t()
                all_logits.append(logits.cpu())
            logits_all = torch.cat(all_logits, dim=0).to(device)
            acc = top1_accuracy(logits_all, torch.tensor(labels, device=device))
            accuracies.append(acc)
        results[name] = accuracies
        plot_degradation(severity_levels, accuracies, f"{name} corruption degradation", out_path=str(Path(results_dir)/f"{name}_degradation.png"))

        # qualitative failures: look at strongest severity
        sev = severity_levels[-1]
        imgs_cor = [fn(img, sev) for img in images]
        batch_size = 64
        failed_images = []
        pred_names = []
        target_names = []
        for i in range(0, len(imgs_cor), batch_size):
            batch = imgs_cor[i:i+batch_size]
            imgs_t = torch.cat([preprocess(im).unsqueeze(0) for im in batch]).to(device)
            with torch.no_grad():
                img_feats = model.encode_image(imgs_t).float()
            img_norm = img_feats / img_feats.norm(dim=1, keepdim=True)
            logits = img_norm @ text_feats.t()
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            lbls = torch.tensor(labels[i:i+len(batch)]).numpy().tolist()
            for j, (p, t) in enumerate(zip(preds, lbls)):
                if p != t and len(failed_images) < 8:
                    failed_images.append(batch[j])
                    pred_names.append(classnames[p])
                    target_names.append(classnames[t])
            if len(failed_images) >= 8:
                break
        if failed_images:
            show_failure_cases(failed_images, pred_names, target_names, classnames, out_path=str(Path(results_dir)/f"{name}_failures.png"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    model, preprocess, device = load_clip(args.device)
    ds = load_dataset(args.data_root)
    os.makedirs(args.results_dir, exist_ok=True)
    res = evaluate_corruptions(ds, model, preprocess, device, args.results_dir)
    print('Corruption experiment results (accuracy per severity):')
    for k,v in res.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
