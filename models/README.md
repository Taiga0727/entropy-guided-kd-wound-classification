# Trained models

The five trained models are published as assets of the [`v1.0` release](https://github.com/Taiga0727/entropy-guided-kd-wound-classification/releases/tag/v1.0) (about 351 MB in total) instead of being committed, so cloning this repository stays small. File names and sizes match the paper's Table VII (sizes there are in MiB, e.g. 83.78).

```bash
# all five files (GitHub CLI), run from the repository root
gh release download v1.0 --repo Taiga0727/entropy-guided-kd-wound-classification --dir models

# or a single file
curl -L -o models/model93.pkl https://github.com/Taiga0727/entropy-guided-kd-wound-classification/releases/download/v1.0/model93.pkl

# verify (SHA-256 of every file is listed in SHA256SUMS.txt)
cd models && sha256sum -c SHA256SUMS.txt
```

| File | What it is | Bytes | Test accuracy | How to load |
|---|---|---|---|---|
| `model93.pkl` | Teacher, ResNet-34 fine-tuned with fastai (`fine_tune(epochs=20, base_lr=0.0015)`), exported with `learn.export` | 87,851,430 | 99.56% | `load_learner` |
| `pruned_model.pkl` | Teacher after 50% L1 unstructured pruning, exported with `learn.export` | 87,862,873 | 99.33% | `load_learner` |
| `student_model.pkl` | ResNet-18 student, standard KD (`state_dict`) | 44,805,618 | 90.89% | `torch.load` into `StudentResNet` |
| `student_model_KD_E_DATA100.pkl` | ResNet-18 student, entropy-guided KD, 70 epochs (`state_dict`) | 44,807,320 | 95.22% | `torch.load` into `StudentResNet` |
| `truncated_model.pth` | Layer-truncated ResNet-34 (`state_dict`), collapsed to one class on the test set | 85,301,102 | 11.11% | `torch.load` into `TruncatedResNet34` |

`StudentResNet` and `TruncatedResNet34` are defined in the notebook (cells 43/56 and 46). Accuracies are the paper's and the notebook's, measured on the leaky split described in the main README.

```python
from fastai.vision.all import load_learner
teacher = load_learner("models/model93.pkl")        # fastai export (pickle): trusted source only

import torch
state = torch.load("models/student_model_KD_E_DATA100.pkl", map_location="cpu", weights_only=True)
# student = StudentResNet(); student.load_state_dict(state)
```

`model93.pkl` and `pruned_model.pkl` are pickled fastai learners; unpickling can run arbitrary code, so only load them from this release or another source you trust. The `state_dict` files load safely with `weights_only=True`.

## Training and terms

- The teacher starts from fastai's default ImageNet-pretrained ResNet-34 and is fine-tuned on the wound images. The ResNet-18 students and the truncated model are created with `pretrained=False`, so they are trained from scratch.
- The weights were trained on the Kaggle Wound Classification dataset (see [`../data/README.md`](../data/README.md); its licence was not verified), and the teacher inherits ImageNet-pretrained weights. The repository's MIT licence covers the code and write-up; check the dataset and ImageNet terms before any commercial use of the weights.
