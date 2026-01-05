"""Constrained experiments runner

Performs:
- Prompt robustness: 4 variants per class (baseline, synonym, shortened, noisy)
- Image corruption: Gaussian noise and JPEG compression with 3 severities

Constraints: CPU-only, max 50 images, no new datasets, outputs in `results/`.
"""
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import datasets

import clip

from utils.visualization import plot_prompt_comparison, plot_degradation


def load_subset(max_per_class=10, classes=None):
    ds = datasets.CIFAR10(root='data', train=False, download=True)
    all_classes = ds.classes
    if classes is None:
        classes = all_classes[:5]
    images = []
    labels = []
    for ci, cls in enumerate(classes):
        idxs = [i for i,(im,lbl) in enumerate(ds) if ds.classes[lbl]==cls]
        chosen = random.sample(idxs, min(max_per_class, len(idxs)))
        for i in chosen:
            images.append(ds[i][0].convert('RGB'))
            labels.append(ci)
    return images, labels, classes


def gen_prompt_variants(classes, prompts_cfg):
    variants = {}
    # baseline
    baseline = [prompts_cfg['base_template'].format(cls=c) for c in classes]
    variants['baseline'] = baseline
    # synonym
    synmap = prompts_cfg.get('synonyms', {})
    syn = [prompts_cfg['base_template'].format(cls=(synmap.get(c,[c])[0] if c in synmap else c)) for c in classes]
    variants['synonym'] = syn
    # shortened
    short = [f'photo of {c}.' for c in classes]
    variants['shortened'] = short
    # noisy (grammatical noise example or punctuation)
    gn = prompts_cfg.get('grammatical_noise_examples', [])
    noisy = [ (gn[0].format(cls=c) if gn else (prompts_cfg['base_template'].format(cls=c)+'!!!')) for c in classes]
    variants['noisy'] = noisy
    return variants


def evaluate_prompt_variants(model, preprocess, device, images, labels, variants, results_dir):
    device = device or 'cpu'
    out = {}
    # encode images once
    imgs_t = torch.cat([preprocess(im).unsqueeze(0).to(device) for im in images])
    with torch.no_grad():
        img_feats = model.encode_image(imgs_t).float()

    for name, prompts in variants.items():
        tokenized = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_feats = model.encode_text(tokenized).float()
        img_n = img_feats / img_feats.norm(dim=1, keepdim=True)
        text_n = text_feats / text_feats.norm(dim=1, keepdim=True)
        logits = img_n @ text_n.t()
        preds = logits.argmax(dim=1).cpu().numpy()
        correct = sum(1 for i,l in enumerate(labels) if preds[i]==l)
        acc = correct / len(labels)
        out[name] = acc

    # plot aggregate comparison
    plot_prompt_comparison(list(out.keys()), [out[k] for k in out], out_path=str(Path(results_dir)/'prompt_aggregate.png'))
    return out


def add_gaussian_noise(img: Image.Image, sigma_px: float):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma_px, arr.shape)
    arrn = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arrn)


def jpeg_compress(img: Image.Image, quality: int):
    import io
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def evaluate_corruptions(model, preprocess, device, images, labels, results_dir):
    device = device or 'cpu'
    # text prompts
    classes = sorted(list(set(labels)))
    # reconstruct class names from labels mapping used earlier is not needed; we'll reuse prompts from load_subset caller
    # For simplicity, regenerate prompts for CIFAR classes via clip default template
    # Use classes names stored in dataset order
    ds = datasets.CIFAR10(root='data', train=False, download=False)
    class_names = ds.classes[:5]
    prompts = [f'a photo of a {c}.' for c in class_names]
    tokenized = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokenized).float()

    # Gaussian sigma mapping (px): light / medium / strong
    sigmas = [0.02*255.0, 0.06*255.0, 0.12*255.0]
    sigma_names = ['light','medium','strong']
    gauss_results = {}
    for name, sigma in zip(sigma_names, sigmas):
        corrs = [add_gaussian_noise(im, sigma) for im in images]
        imgs_t = torch.cat([preprocess(im).unsqueeze(0).to(device) for im in corrs])
        with torch.no_grad():
            feats = model.encode_image(imgs_t).float()
        fn = feats / feats.norm(dim=1, keepdim=True)
        tn = text_feats / text_feats.norm(dim=1, keepdim=True)
        logits = fn @ tn.t()
        preds = logits.argmax(dim=1).cpu().numpy()
        correct = sum(1 for i,l in enumerate(labels) if preds[i]==l)
        acc = correct / len(labels)
        gauss_results[name] = {'acc':acc, 'examples': corrs}
        # save up to 5 failures
        fails = [i for i in range(len(labels)) if preds[i]!=labels[i]]
        for j, idx in enumerate(fails[:5]):
            corrs[idx].save(Path(results_dir)/f'gauss_{name}_fail_{j}.png')

    # JPEG compression qualities
    qualities = [85, 50, 20]
    jpeg_results = {}
    for qname, q in zip(sigma_names, qualities):
        corrs = [jpeg_compress(im, q) for im in images]
        imgs_t = torch.cat([preprocess(im).unsqueeze(0).to(device) for im in corrs])
        with torch.no_grad():
            feats = model.encode_image(imgs_t).float()
        fn = feats / feats.norm(dim=1, keepdim=True)
        tn = text_feats / text_feats.norm(dim=1, keepdim=True)
        logits = fn @ tn.t()
        preds = logits.argmax(dim=1).cpu().numpy()
        correct = sum(1 for i,l in enumerate(labels) if preds[i]==l)
        acc = correct / len(labels)
        jpeg_results[qname] = {'acc':acc, 'examples': corrs}
        fails = [i for i in range(len(labels)) if preds[i]!=labels[i]]
        for j, idx in enumerate(fails[:5]):
            corrs[idx].save(Path(results_dir)/f'jpeg_{qname}_fail_{j}.png')

    # plot accuracy vs severity for both
    levels = [0,1,2]
    gauss_accs = [gauss_results[n]['acc'] for n in sigma_names]
    jpeg_accs = [jpeg_results[n]['acc'] for n in sigma_names]
    plot_degradation(levels, gauss_accs, 'Gaussian noise degradation', out_path=str(Path(results_dir)/'gauss_degradation.png'))
    plot_degradation(levels, jpeg_accs, 'JPEG compression degradation', out_path=str(Path(results_dir)/'jpeg_degradation.png'))
    return gauss_results, jpeg_results


def main():
    random.seed(0)
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    outdir = results_dir/'full_small'
    outdir.mkdir(parents=True, exist_ok=True)

    device = 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)

    images, labels, classes = load_subset(max_per_class=10, classes=None)
    print(f'Loaded {len(images)} images from classes: {classes}')

    prompts_cfg = json.load(open('prompts/prompts.json','r'))
    variants = gen_prompt_variants(classes, prompts_cfg)
    prompt_results = evaluate_prompt_variants(model, preprocess, device, images, labels, variants, outdir)
    print('Prompt robustness results:')
    baseline_acc = prompt_results['baseline']
    for k,v in prompt_results.items():
        drop = baseline_acc - v
        print(f"{k}: acc={v:.3f}, drop_vs_baseline={drop:.3f}")

    gauss_res, jpeg_res = evaluate_corruptions(model, preprocess, device, images, labels, outdir)
    print('Gaussian results:', {k:gauss_res[k]['acc'] for k in gauss_res})
    print('JPEG results:', {k:jpeg_res[k]['acc'] for k in jpeg_res})

    # aggregate plots already saved under outdir
    print('Saved all outputs under', outdir)


if __name__ == '__main__':
    main()
