# Mini AI Pipeline Project: CIFAR-10 Image Classification
**Name:** Your Name  
**Student ID:** Your Student ID  

---

## 1. Introduction

This project demonstrates a small AI pipeline using CIFAR-10 image classification.  
The goal is to practice building a baseline and an improved pipeline using a pre-trained model, and compare their performance.

- **Task description:** Classify CIFAR-10 images into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- **Motivation:** Image classification is a fundamental AI task, suitable for learning AI pipelines without training large models.
- **Input / Output:** Input: 32×32 RGB image. Output: predicted class label.
- **Success criteria:** Measured by classification accuracy on the test set.

---

## 2. Methods

### 2.1 Naïve Baseline
- **Method description:** Predicts classes randomly (or based on simple frequency statistics).
- **Why naïve:** Does not use any learned model or feature representation.
- **Likely failure modes:** Any image with non-uniform pixel distribution or real patterns will likely be misclassified.

### 2.2 AI Pipeline
- **Models used:** Pre-trained ResNet18 (inference only)
- **Pipeline stages:**
  1. Resize input images from 32×32 to 224×224, normalize.
  2. Forward pass through ResNet18.
  3. Argmax of outputs for predicted class.
- **Design choices and justification:** ResNet18 is a widely used image classifier; using pre-trained weights allows fast inference without fine-tuning. Pipeline is simple but sufficient to practice AI workflow.

---

## 3. Experiments

### 3.1 Dataset
- **Source:** CIFAR-10 dataset from torchvision
- **Total examples:** 60,000 (50,000 train / 10,000 test)
- **Train/Test split:** Standard CIFAR-10 split
- **Preprocessing steps:** Resize, convert to tensor, normalize using ImageNet mean/std

### 3.2 Metrics
- Accuracy

### 3.3 Results

| Method           | Accuracy |
|-----------------|---------|
| Naïve Baseline   | 0.0997  |
| AI Pipeline      | 0.0702  |

**Example cases:**  
- Some random images from each class with baseline vs. AI pipeline predictions (optional: add screenshots if available)

---

## 4. Reflection and Limitations

- Baseline performs at random chance (~10%), as expected.
- Pre-trained ResNet18 without fine-tuning performs poorly (~7%) on CIFAR-10 due to domain mismatch with ImageNet.
- Fine-tuning could improve performance significantly.
- Pipeline allows practice of preprocessing, model inference, and evaluation stages.
- Accuracy metric is simple but sufficient for this small-scale pipeline.
- Time and resource constraints prevented fine-tuning and data augmentation.
- Next steps: fine-tune on CIFAR-10 subset, try data augmentation, or experiment with lightweight models.

---

## References
[1] Jason Wei, et al. "Finetuned language models are zero-shot learners." ICLR 2022. [URL](https://openreview.net/forum?id=gEZrGCozdqR)
