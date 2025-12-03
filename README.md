# Mini AI Pipeline Project

## 1. Task Description & Motivation
**Task:** Image classification on CIFAR-10 dataset (10 classes).  
**Motivation:** Practice building a small AI pipeline, compare naive baseline with a pretrained model, and understand the workflow of AI experiments.

**Input/Output:**  
- Input: 32x32 color images (CIFAR-10)  
- Output: Class label from ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

**Success Criteria:** Higher accuracy compared to naive baseline.

---

## 2. Dataset
- **Source:** CIFAR-10 (torchvision)  
- **Train/Test Split:** 50,000 train / 10,000 test  
- **Preprocessing:**  
  - Baseline: None  
  - ResNet18 pipeline: Resize to 224x224, convert to tensor

---

## 3. Methods

### 3.1 Naive Baseline
- Randomly predicts one of 10 classes  
- **Limitations:** Ignores image content, accuracy around 10%  
- **Failure Mode:** Always performs poorly, no learning involved

### 3.2 AI Pipeline
- **Model:** Pretrained ResNet18 (ImageNet weights)  
- **Pipeline:**  
  1. Resize image to 224x224  
  2. Convert to tensor  
  3. Feed into ResNet18 (no fine-tuning)  
  4. Take argmax of outputs as prediction  
- **Limitations:** Domain mismatch (CIFAR-10 vs ImageNet), no fine-tuning, so accuracy is low

---

## 4. Experiments & Results

| Method                     | Accuracy |
|-----------------------------|---------|
| Naive Baseline              | ~0.10   |
| Pretrained ResNet18 (no FT) | ~0.07   |

**Example Cases (predicted vs true):**  
- Random baseline often wrong  
- ResNet18 sometimes misclassifies 'cat' as 'dog' or 'frog' as 'bird'

---

## 5. Reflection & Limitations
- Naive baseline works as expected (~10% accuracy)  
- Pretrained ResNet18 without fine-tuning performs worse than baseline due to domain mismatch  
- Metric (accuracy) captures overall performance but not class-wise errors  
- Future improvements: fine-tuning, data augmentation, using smaller or custom models, top-k accuracy analysis

