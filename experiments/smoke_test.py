"""Lightweight smoke test for CLIP (ViT-B/32).

Verifies:
- CLIP loads on CPU
- Runs a small zero-shot baseline on a subset of CIFAR-10 (15 images, 5 classes)
- Runs one synonym-based prompt perturbation and compares accuracy
- Saves minimal plots to `results/smoke/`

This is intentionally small and fast; it will report missing dependency errors
instead of attempting large installs.
"""
import os
import sys
from pathlib import Path
import random
import torch
from PIL import Image
import numpy as np

def main():
    outdir = Path('results/smoke')
    outdir.mkdir(parents=True, exist_ok=True)

    # Check imports
    try:
        import clip
    except Exception as e:
        print('ERROR: CLIP package not found or failed to import:', e)
        print('Please install requirements (see requirements.txt).')
        sys.exit(2)
    try:
        from torchvision import datasets
        from torchvision.transforms import ToPILImage
    except Exception as e:
        print('ERROR: torchvision or related imports failed:', e)
        sys.exit(2)

    device = 'cpu'
    print('Loading CLIP (ViT-B/32) on', device)
    try:
        model, preprocess = clip.load('ViT-B/32', device=device)
    except Exception as e:
        print('ERROR: CLIP.load failed:', e)
        sys.exit(2)

    # Load small subset of CIFAR-10
    ds = datasets.CIFAR10(root='data', train=False, download=True)
    classes = ds.classes
    # pick 5 classes
    chosen = classes[:5]
    indices = [i for i,(img,lbl) in enumerate(ds) if ds.classes[lbl] in chosen]
    random.seed(0)
    sample_idx = random.sample(indices, min(15, len(indices)))

    images = [ds[i][0].convert('RGB') for i in sample_idx]
    labels = [ds[i][1] for i in sample_idx]

    # Baseline prompts
    prompts = [f'a photo of a {c}.' for c in chosen]
    print('Prompts (baseline):', prompts)

    tokenized = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokenized).float()

    # Encode images in small batches
    img_tensors = [preprocess(im).unsqueeze(0).to(device) for im in images]
    imgs_batch = torch.cat(img_tensors, dim=0)
    with torch.no_grad():
        img_feats = model.encode_image(imgs_batch).float()

    # normalize and compute similarity
    img_norm = img_feats / img_feats.norm(dim=1, keepdim=True)
    text_norm = text_feats / text_feats.norm(dim=1, keepdim=True)
    logits = img_norm @ text_norm.t()
    preds = logits.argmax(dim=1).cpu().numpy()
    # Map true labels to chosen-class indices (if label not in chosen, mark -1)
    chosen_map = {c:i for i,c in enumerate(chosen)}
    labels_mapped = [chosen_map.get(classes[l], -1) for l in labels]
    valid = [i for i,lm in enumerate(labels_mapped) if lm!=-1]
    correct = sum(1 for i in valid if preds[i]==labels_mapped[i])
    acc = correct / len(valid) if valid else 0.0
    print(f'Baseline accuracy on subset ({len(valid)} images):', acc)

    # Perturbation: synonym replace for one class (if present)
    syn_map = {'dog':'canine', 'cat':'feline', 'truck':'lorry', 'car':'automobile'}
    pert_prompts = []
    for c in chosen:
        pert = syn_map.get(c, c)
        pert_prompts.append(f'a photo of a {pert}.')
    print('Prompts (perturbed):', pert_prompts)

    tokenized2 = clip.tokenize(pert_prompts).to(device)
    with torch.no_grad():
        text_feats2 = model.encode_text(tokenized2).float()

    text_norm2 = text_feats2 / text_feats2.norm(dim=1, keepdim=True)
    logits2 = img_norm @ text_norm2.t()
    preds2 = logits2.argmax(dim=1).cpu().numpy()
    correct2 = sum(1 for i in valid if preds2[i]==labels_mapped[i])
    acc2 = correct2 / len(valid) if valid else 0.0
    print(f'Perturbed accuracy on subset: {acc2} (drop {acc-acc2:.3f})')

    # Save a small summary plot: bar of baseline vs perturbed
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4,3))
        plt.bar(['baseline','perturbed'], [acc, acc2])
        plt.ylim(0,1)
        plt.title('Smoke test: accuracy')
        plt.savefig(outdir/'accuracy_bar.png', bbox_inches='tight')
        print('Saved plot to', outdir/'accuracy_bar.png')
    except Exception as e:
        print('Warning: failed to save plot:', e)

    # Save a few failure images
    try:
        fail_imgs = []
        fail_pred = []
        fail_target = []
        for i in valid:
            if preds2[i] != labels_mapped[i]:
                fail_imgs.append(images[i])
                fail_pred.append(chosen[preds2[i]] if preds2[i]<len(chosen) else 'other')
                fail_target.append(classes[labels[i]])
        if fail_imgs:
            # save first 5
            for j,img in enumerate(fail_imgs[:5]):
                img.save(outdir/f'fail_{j}.png')
            print('Saved failure images to', outdir)
        else:
            print('No failures to save for this small sample')
    except Exception as e:
        print('Warning: failed to save failure images:', e)

    print('Smoke test completed successfully.')


if __name__ == '__main__':
    main()
