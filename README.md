<p align="center">
  <img src="src/crisp/img/CRISP_Med_maskot.png" alt="CRISP-Med Mascot" width="400">
</p>

<h1 align="center">
  🧊 CRISP-Med
</h1>

<h3 align="center">
  Continual Retrieval & Indexing System for Medical Image Perception<br>
  <em>train first, freeze, index, retrieve</em>
</h3>

<p align="center">
  <strong>A medical-oriented extension of CRISP for CT-scan, X-ray, MRI slice, ultrasound, and DICOM-based image classification experiments.</strong>
</p>

Repository:

```text
https://github.com/Hokimastah/CRISP-Med
```

---

## 1. Overview

**CRISP-Med** is a medical image adaptation of **CRISP**: Continual Retrieval & Indexing System for Perception.

The original CRISP architecture performs classification through frozen feature extraction, memory-bank indexing, nearest-neighbor retrieval, and voting. CRISP-Med keeps that retrieval-based architecture, but adds medical image handling and an optional **initial backbone training phase** before indexing.

The recommended CRISP-Med workflow is:

```text
Initial Medical Dataset
→ Train / Fine-tune Medical Backbone
→ Save Checkpoint
→ Freeze Backbone
→ Index Embeddings into Memory Bank
→ Top-k Retrieval
→ Weighted / Majority Voting
→ Predicted Class / Unknown
```

This means the model can learn medical-domain features first, then use those learned features as a stable frozen encoder for retrieval-based incremental classification.

> **Important:** CRISP-Med is intended for research and educational machine learning experiments. It is not a clinical diagnosis system.

---

## 2. Main Idea

CRISP-Med has two phases:

### 2.1 Initial Training Phase

The backbone is trained once on the initial medical dataset using a temporary classification head.

```text
Medical Dataset
→ Medical Preprocessing
→ Backbone + Classification Head
→ CrossEntropy Training
→ Save Checkpoint
```

This phase is useful because generic ImageNet features are often weak for medical images such as CT-scan, X-ray, MRI, and ultrasound.

### 2.2 Retrieval / Indexing Phase

After training, the classification head is removed. The backbone is frozen and used only as a feature extractor.

```text
Medical Image / DICOM Slice
→ Medical Preprocessing
→ Frozen Trained Backbone
→ L2-normalized Embedding
→ Memory Bank
→ Vector Index
→ Top-k Retrieval
→ Voting
→ Prediction
```

During indexing and incremental updates, the backbone is **not trained again**. New data is added by extracting embeddings and storing them in the memory bank.

---

## 3. What Makes CRISP-Med Different from CRISP v1?

| Aspect | CRISP v1 | CRISP-Med |
|---|---|---|
| Main target | General image classification | Medical image classification |
| Input | RGB images | PNG/JPG/TIFF/DICOM medical images |
| Preprocessing | Standard image resize/normalize | CT windowing, percentile normalization, grayscale handling |
| Encoder | ResNet / CLIP | Medical ResNet / DenseNet / EfficientNet |
| Initial training | Not included | Supported through `crisp train` |
| Indexing | Frozen encoder → memory bank | Frozen trained medical encoder → memory bank |
| Incremental update | Add embeddings | Add embeddings |
| Diagnosis use | Not applicable | Research only, not clinical diagnosis |

---

## 4. Key Features

- Medical backbone training before indexing
- Frozen encoder during indexing and inference
- Retrieval-based classification
- Incremental add-only memory update
- DICOM support
- CT intensity windowing
- Grayscale-to-RGB conversion
- Medical image normalization
- Top-k nearest-neighbor retrieval
- Weighted voting and majority voting
- Unknown-class detection using similarity threshold
- NumPy, Annoy, and FAISS retrieval backends
- Python API and command-line interface
- Compatible with CRISP-style memory bank and retriever architecture

---

## 5. Supported Medical Inputs

CRISP-Med supports standard 2D image files and DICOM files.

```text
.jpg, .jpeg, .png, .bmp, .webp, .tif, .tiff, .dcm
```

Typical use cases:

| Input Type | Example |
|---|---|
| CT-scan slice | Lung CT, brain CT, abdominal CT |
| X-ray | Chest X-ray, bone X-ray |
| MRI slice | Brain MRI, knee MRI |
| Ultrasound | Organ or fetal ultrasound |
| DICOM | `.dcm` medical image files |

For full 3D CT or MRI volumes, CRISP-Med currently works best when the volume is converted into representative 2D slices or when slices are indexed individually.

---

## 6. Supported Encoders

### 6.1 Medical Encoders

| Encoder | Description |
|---|---|
| `medical_resnet18` | ResNet18-based medical image encoder |
| `medical_resnet34` | ResNet34-based medical image encoder |
| `medical_resnet50` | ResNet50-based medical image encoder |
| `medical_densenet121` | DenseNet121-based medical image encoder |
| `medical_efficientnet_b0` | EfficientNet-B0-based medical image encoder |

Medical encoders can be used in two ways:

1. **Baseline mode**: use ImageNet weights directly.
2. **Recommended mode**: train/fine-tune first using `crisp train`, then index using the saved checkpoint.

### 6.2 General Encoders

| Encoder | Description |
|---|---|
| `resnet18` | Frozen ResNet18 feature extractor |
| `resnet34` | Frozen ResNet34 feature extractor |
| `resnet50` | Frozen ResNet50 feature extractor |
| `resnet101` | Frozen ResNet101 feature extractor |
| `resnet152` | Frozen ResNet152 feature extractor |
| `clip` | Frozen CLIP image encoder using `open_clip_torch` |
| `arcface` | Optional face-specific encoder, if included in the project |

---

## 7. Medical Preprocessing Modes

CRISP-Med includes preprocessing modes for medical image intensity handling.

| Mode | Recommended Use |
|---|---|
| `ct_lung` | Lung CT-scan windowing |
| `ct_soft` | General soft tissue CT windowing |
| `ct_brain` | Brain CT windowing |
| `percentile` | X-ray, MRI, ultrasound, and general grayscale images |
| `minmax` | Simple min-max normalization |
| `none` | Minimal normalization |

Recommended settings:

| Modality | Suggested `intensity_mode` |
|---|---|
| Lung CT | `ct_lung` |
| Brain CT | `ct_brain` |
| General CT | `ct_soft` |
| X-ray | `percentile` |
| MRI slice | `percentile` |
| Ultrasound | `percentile` |

Normalization options:

| Option | Description |
|---|---|
| `imagenet` | ImageNet mean/std normalization |
| `half` | Mean = 0.5, std = 0.5 |
| `none` | No tensor normalization |

Example:

```python
encoder_kwargs={
    "image_size": 224,
    "intensity_mode": "ct_lung",
    "normalize": "imagenet"
}
```

---

## 8. Retrieval Backends

| Retriever | Description |
|---|---|
| `numpy` | Exact brute-force cosine similarity retrieval |
| `annoy` | Approximate nearest-neighbor retrieval using Annoy |
| `faiss` | Dense vector similarity search using FAISS |

For small datasets, `numpy` is simple and reliable.  
For larger medical memory banks, `faiss` is usually recommended.

---

## 9. Voting Methods

| Voting | Description |
|---|---|
| `weighted` | Class score is calculated from the sum of similarity values |
| `majority` | Class score is calculated from the number of retrieved neighbors |

For medical image classification, `weighted` is usually recommended because it considers how close the retrieved medical samples are to the query image.

---

## 10. Installation

### 10.1 Install from GitHub

```bash
pip install git+https://github.com/Hokimastah/CRISP-Med.git
```

### 10.2 Install Locally for Development

```bash
git clone https://github.com/Hokimastah/CRISP-Med.git
cd CRISP-Med
pip install -e .
```

### 10.3 Install with Medical Dependencies

```bash
pip install -e ".[medical]"
```

The `medical` extra should include:

```text
pydicom
```

### 10.4 Install with Optional Retrieval Backends

Install Annoy support:

```bash
pip install -e ".[annoy]"
```

Install FAISS support:

```bash
pip install -e ".[faiss]"
```

Install CLIP support:

```bash
pip install -e ".[clip]"
```

Install all optional dependencies:

```bash
pip install -e ".[all]"
```

### 10.5 FAISS Installation Note

If `faiss-cpu` cannot be installed through `pip`, especially on some Windows environments, use Conda:

```bash
conda install -c pytorch faiss-cpu
```

---

## 11. Dataset Format

CRISP-Med expects a folder-based classification dataset.

```text
dataset_ct/
├── normal/
│   ├── patient001_slice001.dcm
│   ├── patient001_slice002.dcm
│   └── patient002_slice001.dcm
├── pneumonia/
│   ├── patient003_slice001.dcm
│   ├── patient003_slice002.dcm
│   └── patient004_slice001.dcm
└── tumor/
    ├── patient005_slice001.dcm
    └── patient006_slice001.dcm
```

The folder name is automatically used as the class label.

Example:

```text
dataset_ct/normal/patient001_slice001.dcm    → label = normal
dataset_ct/pneumonia/patient003_slice001.dcm → label = pneumonia
dataset_ct/tumor/patient005_slice001.dcm     → label = tumor
```

You can also use exported medical image formats:

```text
dataset_xray/
├── normal/
│   ├── xray001.png
│   └── xray002.png
└── abnormal/
    ├── xray003.png
    └── xray004.png
```

---

## 12. Recommended Workflow: Train → Index → Predict

This is the recommended workflow for CRISP-Med.

### 12.1 Train Medical Backbone

```bash
crisp train \
  --data dataset_ct \
  --encoder medical_resnet50 \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0001 \
  --output checkpoints/medical_resnet50_ct.pth \
  --image-size 224 \
  --intensity-mode ct_lung \
  --normalize imagenet
```

This command trains a medical image classifier and saves a checkpoint containing:

- encoder name
- class mapping
- preprocessing configuration
- classifier state dict
- backbone state dict
- training history

### 12.2 Index Using the Trained Checkpoint

```bash
crisp index \
  --data dataset_ct \
  --output ct_memory.pkl \
  --encoder medical_resnet50 \
  --retriever numpy \
  --checkpoint checkpoints/medical_resnet50_ct.pth
```

During indexing:

```text
checkpoint is loaded
→ classifier head is ignored
→ backbone becomes frozen feature extractor
→ embeddings are stored in memory bank
```

### 12.3 Predict a Query Image

```bash
crisp predict \
  --image test_ct_slice.dcm \
  --memory ct_memory.pkl \
  --encoder medical_resnet50 \
  --retriever numpy \
  --checkpoint checkpoints/medical_resnet50_ct.pth \
  --top-k 7 \
  --threshold 0.55
```

---

## 13. Baseline Workflow: Index Without Training

You can still use CRISP-Med without initial training, but this is only recommended as a baseline.

```bash
crisp index \
  --data dataset_xray \
  --output xray_memory.pkl \
  --encoder medical_densenet121 \
  --retriever numpy \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"percentile","normalize":"imagenet"}'
```

Prediction:

```bash
crisp predict \
  --image test_xray.png \
  --memory xray_memory.pkl \
  --encoder medical_densenet121 \
  --retriever numpy \
  --top-k 5 \
  --threshold 0.60 \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"percentile","normalize":"imagenet"}'
```

---

## 14. Python API Usage

### 14.1 Train Backbone Programmatically

```python
from crisp.trainer import train_medical_backbone

train_medical_backbone(
    data_dir="dataset_ct",
    encoder="medical_resnet50",
    output="checkpoints/medical_resnet50_ct.pth",
    epochs=20,
    batch_size=16,
    lr=1e-4,
    image_size=224,
    intensity_mode="ct_lung",
    normalize="imagenet",
)
```

### 14.2 Index and Predict with Trained Checkpoint

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_resnet50",
    retriever="numpy",
    device="cuda",
    top_k=7,
    voting="weighted",
    encoder_kwargs={
        "checkpoint_path": "checkpoints/medical_resnet50_ct.pth",
        "image_size": 224,
        "intensity_mode": "ct_lung",
        "normalize": "imagenet",
    },
)

clf.add_folder("dataset_ct")
clf.save("ct_memory.pkl")

result = clf.predict(
    "test_ct_slice.dcm",
    threshold=0.55
)

print(result["status"])
print(result["predicted_label"])
print(result["scores"])
print(result["best_similarity"])
```

### 14.3 CPU Usage

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_resnet18",
    retriever="numpy",
    device="cpu",
    top_k=5,
    voting="weighted",
    encoder_kwargs={
        "image_size": 224,
        "intensity_mode": "percentile",
        "normalize": "imagenet",
    },
)

clf.add_folder("dataset_medical")
result = clf.predict("test_image.png")

print(result["predicted_label"])
```

---

## 15. Incremental Medical Learning

CRISP-Med supports add-only incremental learning. New medical samples or new classes can be added without retraining the encoder.

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_resnet50",
    retriever="numpy",
    device="cuda",
    top_k=7,
    voting="weighted",
    encoder_kwargs={
        "checkpoint_path": "checkpoints/medical_resnet50_ct.pth",
        "image_size": 224,
        "intensity_mode": "ct_lung",
        "normalize": "imagenet",
    },
)

# Initial memory
clf.add_folder("dataset_task_1")
clf.save("memory_task_1.pkl")

# Add new medical samples or new classes
clf.add_folder("dataset_task_2")
clf.save("memory_task_2.pkl")

# Predict using updated memory
result = clf.predict("test_slice.dcm")
print(result["predicted_label"])
```

The incremental flow is:

```text
New medical image + label
→ medical preprocessing
→ frozen trained backbone
→ embedding vector
→ append to memory bank
→ rebuild retrieval index
```

No backbone retraining is performed during incremental updates.

---

## 16. Important Rule: Rebuild Memory if Backbone Changes

The memory bank is valid only for the same:

- encoder architecture
- checkpoint
- image size
- intensity preprocessing mode
- normalization mode

If any of these changes, rebuild the memory bank.

Correct:

```text
train once
→ freeze backbone
→ index dataset_task_1
→ add dataset_task_2 using the same frozen backbone
```

Not recommended:

```text
index dataset_task_1
→ train backbone again
→ add dataset_task_2 into old memory
```

If the backbone is retrained, the old memory bank should be rebuilt because old and new embeddings may no longer exist in the same feature space.

---

## 17. Unknown Class Detection

CRISP-Med can mark a query medical image as unknown if the best similarity score is below a selected threshold.

```python
result = clf.predict(
    "test_ct_slice.dcm",
    threshold=0.55
)

print(result["status"])
print(result["predicted_label"])
print(result["best_similarity"])
```

Possible output:

```python
{
    "status": "unknown",
    "predicted_label": null,
    "best_similarity": 0.41
}
```

Threshold values must be tuned empirically for each dataset, encoder, modality, and preprocessing mode.

---

## 18. Command Line Interface

CRISP-Med provides the `crisp` CLI.

### 18.1 Train

```bash
crisp train \
  --data dataset_ct \
  --encoder medical_resnet50 \
  --output checkpoints/medical_resnet50_ct.pth \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0001 \
  --image-size 224 \
  --intensity-mode ct_lung \
  --normalize imagenet
```

Optional training arguments:

```bash
--weight-decay 0.0001
--val-split 0.2
--device cuda
--num-workers 0
--seed 42
--no-pretrained
--class-weighted-loss
```

### 18.2 Index

```bash
crisp index \
  --data dataset_ct \
  --output ct_memory.pkl \
  --encoder medical_resnet50 \
  --retriever numpy \
  --checkpoint checkpoints/medical_resnet50_ct.pth
```

You can also pass preprocessing manually:

```bash
crisp index \
  --data dataset_ct \
  --output ct_memory.pkl \
  --encoder medical_resnet50 \
  --retriever numpy \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"ct_lung","normalize":"imagenet"}'
```

### 18.3 Predict

```bash
crisp predict \
  --image test_ct_slice.dcm \
  --memory ct_memory.pkl \
  --encoder medical_resnet50 \
  --retriever numpy \
  --checkpoint checkpoints/medical_resnet50_ct.pth \
  --top-k 7 \
  --threshold 0.55
```

---

## 19. System Architecture

```text
                 TRAINING PHASE
             ┌────────────────────┐
             │ Medical Dataset     │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Medical Preprocess  │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Backbone + Head     │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Save Checkpoint     │
             └────────────────────┘


                 INDEXING PHASE
             ┌────────────────────┐
             │ Medical Image       │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Medical Preprocess  │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Frozen Backbone     │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Embedding           │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Memory Bank         │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Retriever Index     │
             └────────────────────┘


                 INFERENCE PHASE
             ┌────────────────────┐
             │ Query Image         │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Query Embedding     │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Top-k Retrieval     │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Voting              │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Prediction / Unknown│
             └────────────────────┘
```

---

## 20. How CRISP-Med Works

### 20.1 Medical Image Loading

Input images are loaded from standard image files or DICOM files.

```text
.dcm / .png / .jpg / .tif → image array
```

For DICOM files, the pixel array is extracted and converted into a normalized 2D image representation.

### 20.2 Medical Intensity Normalization

For CT images, pixel values are windowed according to the selected medical mode.

```text
ct_lung  → lung window
ct_soft  → soft tissue window
ct_brain → brain window
```

For X-ray, MRI, and ultrasound, percentile normalization is often more suitable.

### 20.3 Backbone Training

The training command uses a temporary classifier head.

```text
backbone → classifier head → class prediction
```

The classifier head is only needed for supervised training.

### 20.4 Embedding Extraction

During indexing and inference, the classifier head is not used.

```text
preprocessed image → frozen backbone → embedding vector
```

### 20.5 L2 Normalization

The embedding vector is normalized:

```text
embedding = embedding / ||embedding||
```

This makes dot product retrieval equivalent to cosine similarity when vectors are normalized.

### 20.6 Memory Bank

Each stored medical sample contains:

```python
{
    "embedding": vector,
    "label": "class_name",
    "metadata": {
        "path": "dataset_ct/normal/patient001_slice001.dcm"
    }
}
```

### 20.7 Retrieval

During inference, CRISP-Med searches for the most similar stored medical embeddings.

```text
query embedding → top-k nearest neighbors
```

### 20.8 Voting

For weighted voting:

```text
Score(class) = sum(similarity of retrieved neighbors from that class)
```

For majority voting:

```text
Score(class) = number of retrieved neighbors from that class
```

The class with the highest score becomes the prediction, unless the threshold marks the query as unknown.

---

## 21. Suggested Evaluation Metrics

For medical image classification:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- Confusion matrix
- Per-class accuracy

For retrieval-based experiments:

- Top-1 retrieval accuracy
- Top-k retrieval accuracy
- Mean reciprocal rank
- Mean average precision
- Best similarity distribution
- Threshold sensitivity analysis

For incremental learning experiments:

- Average accuracy across tasks
- Forgetting score
- Backward transfer
- Forward transfer
- Per-task accuracy
- Memory size per task
- Accuracy-memory trade-off

For medical datasets, report class imbalance clearly and include per-class metrics, not only overall accuracy.

---

## 22. Recommended Dataset Splitting

For medical imaging, avoid data leakage.

Recommended split:

```text
patient-level split
not slice-level split
```

This means slices from the same patient should not appear in both training memory and test data.

Bad split:

```text
patient001_slice001 → memory
patient001_slice002 → test
```

Good split:

```text
patient001_all_slices → memory
patient002_all_slices → test
```

Patient-level splitting is especially important for CT, MRI, and repeated X-ray studies.

---

## 23. Project Structure

```text
CRISP-Med/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/
│   └── architecture.md
├── examples/
│   ├── medical_ctscan_usage.py
│   ├── medical_xray_usage.py
│   └── train_then_index.py
├── tests/
│   ├── test_memory.py
│   ├── test_voting.py
│   └── test_medical_preprocessing.py
└── src/
    └── crisp/
        ├── __init__.py
        ├── classifier.py
        ├── cli.py
        ├── datasets.py
        ├── memory.py
        ├── trainer.py
        ├── utils.py
        ├── voting.py
        ├── encoders/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── factory.py
        │   ├── resnet.py
        │   ├── clip_encoder.py
        │   ├── arcface.py
        │   └── medical.py
        └── retrievers/
            ├── __init__.py
            ├── base.py
            ├── factory.py
            ├── numpy_backend.py
            ├── annoy_backend.py
            └── faiss_backend.py
```

---

## 24. API Reference

### 24.1 `CRISPClassifier`

```python
CRISPClassifier(
    encoder="medical_resnet50",
    retriever="numpy",
    pretrained=True,
    device=None,
    top_k=5,
    voting="weighted",
    encoder_kwargs=None,
    retriever_kwargs=None
)
```

### Parameters

| Parameter | Description |
|---|---|
| `encoder` | Encoder name, for example `medical_resnet50`, `medical_densenet121`, `resnet50`, or `clip` |
| `retriever` | Retrieval backend: `numpy`, `annoy`, or `faiss` |
| `pretrained` | Whether to use pretrained weights when supported |
| `device` | `cuda`, `cpu`, or `None` for automatic selection |
| `top_k` | Number of nearest neighbors used for voting |
| `voting` | `weighted` or `majority` |
| `encoder_kwargs` | Additional arguments for the selected encoder |
| `retriever_kwargs` | Additional arguments for the selected retriever |

### 24.2 Medical `encoder_kwargs`

| Argument | Description | Example |
|---|---|---|
| `image_size` | Input image size | `224` |
| `intensity_mode` | Medical preprocessing mode | `ct_lung` |
| `normalize` | Normalization mode | `imagenet` |
| `checkpoint_path` | Trained medical checkpoint | `checkpoints/model.pth` |

### 24.3 Main Methods

```python
clf.add_image(image_path, label)
```

Add one labeled medical image to the memory bank.

```python
clf.add_folder(folder)
```

Add a folder dataset to the memory bank.

```python
clf.predict(image_path, top_k=None, threshold=None)
```

Predict the class of a query medical image.

```python
clf.save(path)
```

Save memory bank to a `.pkl` file.

```python
clf.load(path)
```

Load memory bank from a `.pkl` file.

---

## 25. Output Format

Example prediction output:

```python
{
    "status": "known",
    "predicted_label": "pneumonia",
    "scores": {
        "pneumonia": 4.12,
        "normal": 1.08
    },
    "best_similarity": 0.87,
    "neighbors": [
        {
            "index": 0,
            "label": "pneumonia",
            "similarity": 0.87,
            "metadata": {
                "path": "dataset_xray/pneumonia/patient003.png"
            }
        }
    ],
    "encoder": "medical_densenet121",
    "retriever": "numpy"
}
```

---

## 26. Notes and Limitations

- CRISP-Med is intended for research and education, not direct clinical diagnosis.
- The backbone may be trained before indexing, but it is frozen during indexing and inference.
- If the encoder, checkpoint, image size, intensity mode, or normalization mode changes, the memory bank should be rebuilt.
- Threshold values must be tuned for each dataset and modality.
- DICOM images may have modality-specific metadata and intensity scaling; verify preprocessing before evaluation.
- For 3D CT/MRI, CRISP-Med currently operates on 2D slices unless a 3D encoder is added.
- Slice-level splitting can cause data leakage. Use patient-level splitting whenever possible.
- ImageNet-pretrained encoders are only baselines. Better performance usually requires medical-domain training or fine-tuning before freezing.
- Retrieval memory may grow large if all slices are stored. Use representative slice selection or prototype selection when needed.
- Medical datasets are often imbalanced. Report sensitivity, specificity, F1-score, and ROC-AUC in addition to accuracy.

---

## 27. Roadmap

Planned improvements:

- 3D CT/MRI encoder support
- Patient-level aggregation
- Multi-slice voting
- Modality-aware memory bank
- Organ-specific routing
- Medical-domain pretrained checkpoints
- Grad-CAM or retrieval explanation interface
- Better DICOM metadata handling
- Prototype selection per patient and per class
- Evaluation command for full test folders

---

## 28. License

This project is released under the MIT License.

---

## 29. Citation

If you use CRISP-Med in an academic project, you can cite this repository as:

```bibtex
@software{crispmed2026,
  title = {CRISP-Med: Continual Retrieval and Indexing System for Medical Image Perception},
  author = {Satrio Puji Danuirto},
  year = {2026},
  url = {https://github.com/Hokimastah/CRISP-Med}
}
```

---

## 30. Disclaimer

CRISP-Med does not provide medical advice, diagnosis, or treatment recommendations.  
All outputs should be interpreted only as machine learning experiment results and must not be used as a replacement for professional medical judgment.
