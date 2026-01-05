# CLIP Robustness Analysis (ViT-B/32)

Project goal: evaluate zero-shot robustness and failure modes of CLIP (ViT-B/32) to prompt perturbations, image corruptions, and domain shifts. This repository implements compact, reproducible experiments suitable for a research internship-style report.

**Key ideas and motivation**
- CLIP is widely used for zero-shot image classification but can be brittle to small changes in prompts and input images. Understanding these failure modes is important for robustness and safety.

**Setup (lightweight)**
1. Create a Python venv and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Data: scripts download CIFAR-10 automatically to `./data`.

**Project structure**
- `data/` — datasets (downloaded by scripts)
- `prompts/` — prompt templates and synonym maps
- `experiments/` — three experiment drivers:
  - `prompt_robustness.py` — prompt perturbations
  - `image_corruption.py` — controlled corruptions across severity
  - `distribution_shift.py` — simulate domain shift
- `utils/` — helper modules: `model.py`, `metrics.py`, `visualization.py`
- `results/` — figures and outputs

**Experimental setup (concise)**
- Model: CLIP ViT-B/32 (zero-shot) using OpenAI implementation.
- Dataset: CIFAR-10 (small, quick to run); we simulate domain shift by applying strong image transforms to create Dataset B.
- Metrics: top-1 accuracy, mean cosine similarity between image and correct class text features. Visualizations: accuracy vs severity, prompt comparison bar plots, failure-case gallery.

**How to run**
- Prompt robustness:

  ```bash
  python experiments/prompt_robustness.py --data_root ./data --results_dir ./results
  ```

- Image corruptions:

  ```bash
  python experiments/image_corruption.py --data_root ./data --results_dir ./results
  ```

- Distribution shift:

  ```bash
  python experiments/distribution_shift.py --data_root ./data --results_dir ./results
  ```

**Key findings (expected / research notes)**
- Prompt phrasing and synonym choice can change CLIP scores and accuracy noticeably; templating and ensembling text prompts mitigate some brittleness.
- Low-to-moderate image corruptions (noise, compression) steadily degrade performance; motion blur and heavy JPEG often cause larger drops.
- Simulated domain shifts (color/texture changes) lead to measurable accuracy drops even when classes overlap, highlighting distributional sensitivity.

**Limitations**
- Experiments use CIFAR-10 and synthetic transforms; larger, real-world domain shifts (e.g., artwork vs. photos) require more data.
- No heavy adversarial attacks here; lightweight FGSM-style perturbations can be added in future work.

**Future work**
- Evaluate prompt ensembling and calibrated prompt engineering for robustness.
- Test on larger datasets and real distribution shifts (ImageNet → ImageNetV2, artwork datasets).
- Explore contrastive finetuning and robust pretraining to improve alignment between modalities.

**Ethics and safety note**
Understanding model failure modes is essential for safe deployment. This repository aims to provide reproducible, small-scale experiments to guide principled analysis; it is not intended for misuse.

**Results (summary)**
- **Prompt robustness:** On a constrained 5-class CIFAR-10 subset (≤50 images) the evaluated prompt variants produced the following example metrics: baseline acc=0.920, synonym acc=0.920 (no drop), shortened acc=0.960 (improvement), noisy acc=0.940. See aggregate comparison: [prompt_aggregate.png](results/full_small/prompt_aggregate.png).
- **Image corruption:** Gaussian noise and JPEG compression produce steady degradation. Example accuracies (light/medium/strong): Gaussian ≈ [0.86, 0.70, 0.44]; JPEG ≈ [0.82, 0.74, 0.48]. See plots: [gauss_degradation.png](results/full_small/gauss_degradation.png) and [jpeg_degradation.png](results/full_small/jpeg_degradation.png).
- **Key quantitative numbers:** baseline vs corrupted (sample): baseline acc≈0.92 → Gaussian strong acc≈0.44 (≈52% relative drop), JPEG strong acc≈0.48 (≈48% relative drop) on the small-sample run.

**Limitations & Scope**
- Small-sample evaluation: experiments used a limited CIFAR-10 subset (first 5 classes, ≤50 images) for fast, reproducible checks. Results are illustrative, not definitive.
- CPU-only inference: all experiments were run on CPU to ensure reproducibility on low-resource machines; wall-time is higher than GPU runs.
- No adversarial optimization: only lightweight/noise-style corruptions and simple prompt perturbations were used; no adversarial attack libraries or optimization loops were run.

**Reproducibility**
- To re-run the constrained experiments (same setup used for the figures):

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  # ensure repo root is on PYTHONPATH so local utils import works
  PYTHONPATH=. python experiments/run_small_experiments.py
  ```

- Expected runtime (rough):
  - First run (weights not cached): 5–12 minutes (downloads model weights ~0.5GB total and runs CPU inference).
  - Subsequent runs (cached weights): ~1–4 minutes on CPU for the constrained (≤50-image) experiments.

- Cached weight behavior: CLIP model weights are downloaded on first call to `clip.load(...)` and cached by the CLIP package; subsequent runs reuse the cached files and avoid re-downloading weights.

**Where outputs are saved**
- All experiment outputs and plots for the constrained run are under `results/full_small/` (figures `prompt_aggregate.png`, `gauss_degradation.png`, `jpeg_degradation.png`, and several failure example images).

# clip-robustness-analysis