"""
Prompt robustness experiments for CLIP (ViT-B/32).

Runs zero-shot classification on a small dataset (CIFAR-10) and measures
performance under prompt perturbations: synonyms, length variation,
grammatical noise, and random deletion.

Designed for research-oriented reproducibility and readability.
"""
import argparse
import json
import random
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from torchvision import datasets
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

from utils.model import load_clip, pil_to_clip
from utils.metrics import top1_accuracy, mean_cosine_similarity
from utils.visualization import plot_prompt_comparison, show_failure_cases


def load_dataset(root, split='test'):
    # CIFAR-10 is a small, realistic dataset for quick experiments
    ds = datasets.CIFAR10(root=root, train=(split=='train'), download=True)
    return ds


def build_text_prompts(classnames, templates):
    prompts = []
    for name in classnames:
        prompts.append(templates['base_template'].format(cls=name))
    return prompts


def evaluate_prompts(prompts_list, dataset, model, preprocess, device, results_dir):
    # Precompute text features for each prompt and class label
    texts_features = []
    for prompt in prompts_list:
        tokenized = torch.cat([torch.tensor(0)]) if False else None
    # We'll use CLIP tokenization per prompt set below

    images = []
    labels = []
    for img, label in dataset:
        images.append(img)
        labels.append(label)

    # Encode images in batches to avoid large memory spikes
    batch_size = 64
    feats = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i+batch_size]
        batch_t = torch.cat([pil_to_clip(im, preprocess) for im in batch_imgs]).to(device)
        with torch.no_grad():
            batch_f = model.encode_image(batch_t).float().cpu()
        feats.append(batch_f)
    image_feats = torch.cat(feats, dim=0).to(device)

    results = {}
    for name, prompts in prompts_list.items():
        # prompts: list of strings, one per class name
        tokenized = torch.cat([torch.tensor(0)]) if False else None
        try:
            import clip
            tokenized = clip.tokenize(prompts).to(device)
            with torch.no_grad():
                text_feats = model.encode_text(tokenized).float()
        except Exception:
            raise RuntimeError('CLIP tokenize/encode failed; ensure CLIP is installed')

        # text_feats are per-class features; compute similarity for each image
        # Normalize
        image_norm = image_feats / image_feats.norm(dim=1, keepdim=True)
        text_norm = text_feats / text_feats.norm(dim=1, keepdim=True)

        logits = (image_norm @ text_norm.t())
        labels_tensor = torch.tensor(labels, device=device)
        acc = top1_accuracy(logits, labels_tensor)
        mean_sim = mean_cosine_similarity(image_feats, text_feats[labels_tensor])
        # record basic metrics
        results[name] = {'accuracy': acc, 'mean_cosine': mean_sim}

        # save qualitative failure cases for this prompt type
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        lbls = labels_tensor.cpu().numpy().tolist()
        # collect first up to 8 failures
        failed_images = []
        pred_names = []
        target_names = []
        for i, (p,t) in enumerate(zip(preds, lbls)):
            if p != t and len(failed_images) < 8:
                failed_images.append(images[i])
                pred_names.append(dataset.classes[p])
                target_names.append(dataset.classes[t])
        if failed_images:
            show_failure_cases(failed_images, pred_names, target_names, dataset.classes, out_path=str(Path(results_dir)/f"fail_{name}.png"))

    # Visualization: barplot of prompt accuracies
    plot_prompt_comparison(list(results.keys()), [r['accuracy'] for r in results.values()], out_path=str(Path(results_dir)/"prompt_comparison.png"))

    # Save numeric results and relative drop vs baseline
    try:
        import json, os
        os.makedirs(results_dir, exist_ok=True)
        base_acc = results.get('base', {}).get('accuracy', None)
        serial = {}
        for k, v in results.items():
            rel_drop = None
            if base_acc is not None and base_acc > 0:
                rel_drop = (base_acc - v['accuracy']) / base_acc
            serial[k] = {'accuracy': v['accuracy'], 'mean_cosine': v['mean_cosine'], 'relative_drop': rel_drop}
        with open(Path(results_dir)/'prompt_results.json', 'w') as f:
            json.dump(serial, f, indent=2)
    except Exception:
        pass

    return results


def generate_perturbations(classnames, prompts_cfg):
    # Build several prompt variants for all classes
    variants = {}
    # base
    base = [prompts_cfg['base_template'].format(cls=c) for c in classnames]
    variants['base'] = base

    # synonym-based: use first available synonym per class (if any), else keep original
    synmap = prompts_cfg.get('synonyms', {})
    syn_variant = []
    for c in classnames:
        if c in synmap and len(synmap[c]) > 0:
            syn_variant.append(prompts_cfg['base_template'].format(cls=synmap[c][0]))
        else:
            syn_variant.append(prompts_cfg['base_template'].format(cls=c))
    variants['synonym'] = syn_variant

    # shortened prompt: prefer a short grammatical noise example if present (e.g., 'photo of {cls}')
    short_template = None
    for ex in prompts_cfg.get('grammatical_noise_examples', []):
        if ex.strip().lower().startswith('photo of'):
            short_template = ex
            break
    if short_template is None:
        short_template = 'photo of {cls}'
    variants['short'] = [short_template.format(cls=c) for c in classnames]

    # noisy / word-dropped: random single-token deletion per class
    rnd = []
    for c in classnames:
        words = prompts_cfg['base_template'].format(cls=c).split()
        if len(words) > 1:
            del words[random.randrange(0, len(words))]
        rnd.append(' '.join(words))
    variants['random_delete'] = rnd
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data', help='Path to store datasets')
    parser.add_argument('--results_dir', default='./results', help='Where to save figures')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess, device = load_clip(device)

    ds = load_dataset(args.data_root, split='test')
    classnames = ds.classes
    # load prompt config
    cfg = json.load(open('prompts/prompts.json', 'r'))
    variants = generate_perturbations(classnames, cfg)

    results = evaluate_prompts(variants, ds, model, preprocess, device, args.results_dir)
    print('Prompt robustness results:')
    for k,v in results.items():
        print(f"{k}: acc={v['accuracy']:.3f}, mean_cosine={v['mean_cosine']:.3f}")


if __name__ == '__main__':
    main()
