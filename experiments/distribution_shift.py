"""
Distribution shift analysis: Train/evaluate on Dataset A and test on Dataset B
with overlapping classes. For a lightweight, reproducible experiment we simulate
a domain shift by applying strong style transforms to CIFAR-10 to create Dataset B.
"""
import argparse
import os
import torch
from torchvision import transforms, datasets
from pathlib import Path

from utils.model import load_clip, pil_to_clip
from utils.metrics import top1_accuracy
from utils.visualization import plot_degradation


def make_shifted_dataset(root, severity=1):
    # use CIFAR-10 and create a transformed copy as shifted domain
    base = datasets.CIFAR10(root=root, train=False, download=True)
    # define shift transforms (strong color jitter + grayscale + resize jitter)
    tf = transforms.Compose([
        transforms.ColorJitter(brightness=0.5*severity, contrast=0.5*severity, saturation=0.5*severity),
        transforms.RandomGrayscale(p=0.3),
    ])
    images = []
    labels = []
    for img, lbl in base:
        images.append(tf(img).convert('RGB'))
        labels.append(lbl)
    return images, labels, base.classes


def evaluate_shift(model, preprocess, device, imagesA, labelsA, imagesB, labelsB, classes):
    import clip
    text_prompts = [f"a photo of a {c}." for c in classes]
    tokenized = clip.tokenize(text_prompts).to(device)
    with torch.no_grad():
        text_feats = model.encode_text(tokenized).float()

    def encode_images(imgs):
        batch = torch.cat([preprocess(im).unsqueeze(0) for im in imgs]).to(device)
        with torch.no_grad():
            return model.encode_image(batch).float()

    featsA = encode_images(imagesA)
    featsB = encode_images(imagesB)

    imgA_norm = featsA / featsA.norm(dim=1, keepdim=True)
    imgB_norm = featsB / featsB.norm(dim=1, keepdim=True)
    text_norm = text_feats / text_feats.norm(dim=1, keepdim=True)

    logitsA = imgA_norm @ text_norm.t()
    logitsB = imgB_norm @ text_norm.t()

    accA = top1_accuracy(logitsA, torch.tensor(labelsA, device=device))
    accB = top1_accuracy(logitsB, torch.tensor(labelsB, device=device))
    return accA, accB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--results_dir', default='./results')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    model, preprocess, device = load_clip(args.device)
    # Dataset A: original CIFAR-10 test
    base = datasets.CIFAR10(root=args.data_root, train=False, download=True)
    imagesA = [img.convert('RGB') for img,_ in base]
    labelsA = [lbl for _,lbl in base]

    # Dataset B: shifted (vary severity)
    severities = [1,2,3]
    results = {}
    for s in severities:
        imagesB, labelsB, classes = make_shifted_dataset(args.data_root, severity=s)
        accA, accB = evaluate_shift(model, preprocess, device, imagesA, labelsA, imagesB, labelsB, classes)
        results[s] = {'acc_A': accA, 'acc_B': accB, 'drop': accA - accB}

    os.makedirs(args.results_dir, exist_ok=True)
    # plot drop vs severity
    plot_degradation(list(results.keys()), [results[s]['drop'] for s in results], 'Distribution shift drop', out_path=str(Path(args.results_dir)/'distribution_shift_drop.png'))
    print('Distribution shift results (A=original, B=shifted):')
    for s,v in results.items():
        print(f"severity {s}: acc_A={v['acc_A']:.3f}, acc_B={v['acc_B']:.3f}, drop={v['drop']:.3f}")


if __name__ == '__main__':
    main()
