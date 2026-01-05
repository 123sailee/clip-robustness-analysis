import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

def plot_degradation(severity_levels, accuracies, title, out_path=None):
    plt.figure(figsize=(6,4))
    plt.plot(severity_levels, accuracies, marker='o')
    plt.xlabel('Severity')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0,1)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

def plot_prompt_comparison(prompt_names, accuracies, out_path=None):
    plt.figure(figsize=(8,4))
    sns.barplot(x=prompt_names, y=accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Prompt')
    plt.ylim(0,1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

def show_failure_cases(images, predicted, target, labels, out_path=None):
    # images: list of PIL images; predicted/target/labels are lists of strings
    n = min(8, len(images))
    plt.figure(figsize=(12,4))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(images[i])
        plt.title(f"P:{predicted[i]}\nT:{target[i]}")
        plt.axis('off')
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()
