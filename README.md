<p align="center">
  <img src="src/crisp/img/CRISP_Med_maskot.png" alt="CRISP-Med Mascot" width="400">
</p>

<h1 align="center">
  🧊 CRISP-Med
</h1>

<h3 align="center">
  Continual Retrieval & Indexing System for Medical Image Perception<br>
  <em>retrieval-based medical image classification without catastrophic forgetting</em>
</h3>

<p align="center">
  <strong>A modular, medical-oriented extension of CRISP for CT-scan, X-ray, MRI slice, ultrasound, and DICOM-based image classification experiments.</strong>
</p>

Repository:

```text
https://github.com/Hokimastah/CRISP-Med
```

---

## 1. Overview

**CRISP-Med** is a medical image adaptation of **CRISP**: Continual Retrieval & Indexing System for Perception.

Instead of retraining the model every time new medical samples or new disease classes are added, CRISP-Med extracts image embeddings using a frozen medical-oriented encoder, stores them in a memory bank, and classifies query images through top-k similarity retrieval and voting.

The core idea is:

```text
Medical Image / DICOM Slice
→ Medical Preprocessing
→ Frozen Medical Encoder
→ Feature Embedding
→ Memory Bank
→ Vector Index
→ Top-k Retrieval
→ Weighted / Majority Voting
→ Predicted Class
```

CRISP-Med is designed for research and experimental use in:

- CT-scan slice classification
- X-ray image classification
- MRI slice classification
- Ultrasound image classification
- DICOM-to-embedding retrieval
- Incremental medical image classification
- Retrieval-augmented medical image analysis
- Medical image prototype memory experiments

> **Important:** CRISP-Med is not a clinical diagnosis system. It is intended for research, educational, and experimental machine learning use only.

---

## 2. Why CRISP-Med?

Standard CRISP uses general-purpose image encoders such as ResNet or CLIP. These encoders are useful as baselines, but medical images have different visual characteristics from natural images:

- CT and MRI are often grayscale.
- DICOM images may contain intensity values that need windowing.
- Medical classes can differ through subtle texture, density, shape, or lesion patterns.
- A single patient may have many slices, creating large memory banks.
- Incremental datasets may introduce new disease classes, organs, or modalities over time.

CRISP-Med addresses these issues by adding:

- Medical image preprocessing
- DICOM reading support
- CT windowing modes
- Grayscale-to-RGB adaptation
- Medical encoder options
- Optional herding-based memory reduction
- Retrieval-based classification for expandable class memory

---

## 3. Main Features

- Frozen medical image encoder
- Retrieval-based classification
- Incremental add-only memory update
- DICOM image support
- CT intensity windowing
- Grayscale image support
- Top-k nearest neighbor retrieval
- Weighted voting and majority voting
- Optional unknown-class detection through similarity threshold
- Optional herding / prototype memory compression
- NumPy, Annoy, and FAISS retrieval backends
- Python API and command-line interface
- Installable as a Python package

---

## 4. Supported Medical Inputs

CRISP-Med supports common 2D image formats and DICOM files.

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

## 5. Supported Encoders

### 5.1 General Encoders

| Encoder | Description |
|---|---|
| `resnet18` | Frozen ResNet18 feature extractor |
| `resnet34` | Frozen ResNet34 feature extractor |
| `resnet50` | Frozen ResNet50 feature extractor |
| `resnet101` | Frozen ResNet101 feature extractor |
| `resnet152` | Frozen ResNet152 feature extractor |
| `clip` | Frozen CLIP image encoder using `open_clip_torch` |

### 5.2 Medical-Oriented Encoders

| Encoder | Description |
|---|---|
| `medical_resnet18` | ResNet18-based medical image embedding encoder |
| `medical_resnet34` | ResNet34-based medical image embedding encoder |
| `medical_resnet50` | ResNet50-based medical image embedding encoder |
| `medical_densenet121` | DenseNet121-based medical image embedding encoder |
| `medical_efficientnet_b0` | EfficientNet-B0-based medical image embedding encoder |

Medical encoders can be used with ImageNet weights as a baseline or with a custom checkpoint trained on medical images.

Example with a custom checkpoint:

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_resnet50",
    retriever="numpy",
    device="cuda",
    encoder_kwargs={
        "checkpoint_path": "checkpoints/medical_resnet50.pth",
        "image_size": 224,
        "intensity_mode": "ct_lung"
    }
)
```

---

## 6. Medical Preprocessing Modes

CRISP-Med includes preprocessing modes for medical images.

| Mode | Use Case |
|---|---|
| `ct_lung` | Lung CT-scan windowing |
| `ct_soft` | Soft tissue CT windowing |
| `ct_brain` | Brain CT windowing |
| `percentile` | Robust normalization using percentile clipping |
| `minmax` | Min-max normalization |
| `none` | Basic image loading without medical-specific windowing |

Example:

```python
encoder_kwargs={
    "intensity_mode": "ct_lung",
    "image_size": 224,
    "normalize": "imagenet"
}
```

Recommended settings:

| Modality | Suggested `intensity_mode` |
|---|---|
| Lung CT | `ct_lung` |
| Brain CT | `ct_brain` |
| General CT | `ct_soft` |
| X-ray | `percentile` |
| MRI slice | `percentile` |
| Ultrasound | `percentile` |

---

## 7. Retrieval Backends

| Retriever | Description |
|---|---|
| `numpy` | Exact brute-force cosine similarity retrieval |
| `annoy` | Approximate nearest neighbor retrieval using Annoy |
| `faiss` | Dense vector similarity search using FAISS |

For small datasets, `numpy` is simple and reliable.  
For larger medical memory banks, `faiss` is usually recommended.

---

## 8. Voting Methods

| Voting | Description |
|---|---|
| `weighted` | Class score is calculated from the sum of similarity values |
| `majority` | Class score is calculated from the number of retrieved neighbors |

For medical image classification, `weighted` is usually recommended because it considers how close the retrieved medical samples are to the query image.

---

## 9. Installation

### 9.1 Install from GitHub

```bash
pip install git+https://github.com/Hokimastah/CRISP-Med.git
```

### 9.2 Install Locally for Development

```bash
git clone https://github.com/Hokimastah/CRISP-Med.git
cd CRISP-Med
pip install -e .
```

### 9.3 Install with Medical Dependencies

```bash
pip install -e ".[medical]"
```

### 9.4 Install with Optional Retrieval Backends

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

### 9.5 FAISS Installation Note

If `faiss-cpu` cannot be installed through `pip`, especially on some Windows environments, use Conda:

```bash
conda install -c pytorch faiss-cpu
```

---

## 10. Dataset Format

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

## 11. Basic Python Usage

### 11.1 CT-scan Slice Classification

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_resnet50",
    retriever="numpy",
    device="cuda",
    top_k=7,
    voting="weighted",
    encoder_kwargs={
        "image_size": 224,
        "intensity_mode": "ct_lung",
        "normalize": "imagenet"
    }
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

### 11.2 Chest X-ray Classification

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_densenet121",
    retriever="faiss",
    device="cuda",
    top_k=5,
    voting="weighted",
    encoder_kwargs={
        "image_size": 224,
        "intensity_mode": "percentile",
        "normalize": "imagenet"
    }
)

clf.add_folder("dataset_xray")
clf.save("xray_memory.pkl")

result = clf.predict("test_xray.png")

print(result)
```

### 11.3 CPU Usage

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
        "intensity_mode": "percentile"
    }
)

clf.add_folder("dataset_medical")
result = clf.predict("test_image.png")

print(result["predicted_label"])
```

---

## 12. Incremental Medical Learning

CRISP-Med supports add-only incremental learning. New medical samples or new classes can be added without retraining the encoder.

```python
from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_resnet50",
    retriever="faiss",
    device="cuda",
    top_k=7,
    voting="weighted",
    encoder_kwargs={
        "image_size": 224,
        "intensity_mode": "ct_soft"
    }
)

# Initial memory
clf.add_folder("dataset_task_1")
clf.save("memory_task_1.pkl")

# Add new disease class or new samples
clf.add_folder("dataset_task_2")
clf.save("memory_task_2.pkl")

# Predict using updated memory
result = clf.predict("test_slice.dcm")
print(result["predicted_label"])
```

The flow is:

```text
New medical image + label
→ medical preprocessing
→ frozen medical encoder
→ embedding vector
→ append to memory bank
→ rebuild retrieval index
```

No backbone retraining is performed during incremental updates.

---

## 13. Memory Compression with Herding

Medical datasets can contain many slices per patient. If every slice is stored, the memory bank may become very large and redundant.

CRISP-Med supports prototype selection through herding:

```python
from crisp.prototypes import apply_herding

clf.add_folder("dataset_ct")

apply_herding(
    memory=clf.memory,
    max_per_class=30
)

clf.retriever.build(clf.memory)
clf.save("ct_memory_herding.pkl")
```

CLI example:

```bash
crisp index \
  --data dataset_ct \
  --output ct_memory.pkl \
  --encoder medical_resnet50 \
  --retriever faiss \
  --max-per-class 30 \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"ct_lung","normalize":"imagenet"}'
```

Herding keeps the most representative embeddings per class and reduces redundant memory entries.

---

## 14. Unknown Class Detection

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
    "predicted_label": None,
    "best_similarity": 0.41
}
```

Threshold values must be tuned empirically for each dataset, encoder, modality, and preprocessing mode.

---

## 15. Command Line Interface

CRISP-Med provides the `crisp` CLI.

### 15.1 Build a CT Memory Bank

```bash
crisp index \
  --data dataset_ct \
  --output ct_memory.pkl \
  --encoder medical_resnet50 \
  --retriever numpy \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"ct_lung","normalize":"imagenet"}'
```

### 15.2 Build an X-ray Memory Bank with FAISS

```bash
crisp index \
  --data dataset_xray \
  --output xray_memory.pkl \
  --encoder medical_densenet121 \
  --retriever faiss \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"percentile","normalize":"imagenet"}'
```

### 15.3 Build Memory with Herding

```bash
crisp index \
  --data dataset_ct \
  --output ct_memory_herding.pkl \
  --encoder medical_resnet50 \
  --retriever faiss \
  --top-k 7 \
  --max-per-class 30 \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"ct_soft","normalize":"imagenet"}'
```

### 15.4 Predict a DICOM Slice

```bash
crisp predict \
  --image test_ct_slice.dcm \
  --memory ct_memory.pkl \
  --encoder medical_resnet50 \
  --retriever numpy \
  --top-k 7 \
  --threshold 0.55 \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"ct_lung","normalize":"imagenet"}'
```

### 15.5 Predict an X-ray Image

```bash
crisp predict \
  --image test_xray.png \
  --memory xray_memory.pkl \
  --encoder medical_densenet121 \
  --retriever faiss \
  --top-k 5 \
  --threshold 0.60 \
  --encoder-kwargs '{"image_size":224,"intensity_mode":"percentile","normalize":"imagenet"}'
```

---

## 16. System Architecture

```text
Input Medical Image
      │
      ▼
Medical Image Loader
(PNG/JPG/TIFF/DICOM)
      │
      ▼
Medical Preprocessing
(CT Windowing / Percentile / Min-Max)
      │
      ▼
Frozen Medical Encoder
(ResNet / DenseNet / EfficientNet)
      │
      ▼
L2-Normalized Embedding
      │
      ▼
Memory Bank
(Embedding + Label + Metadata)
      │
      ▼
Retriever
(NumPy / Annoy / FAISS)
      │
      ▼
Top-k Similar Samples
      │
      ▼
Voting
(Weighted / Majority)
      │
      ▼
Predicted Class / Unknown
```

---

## 17. How CRISP-Med Works

### 17.1 Medical Image Loading

Input images are loaded from standard image files or DICOM files.

```text
.dcm / .png / .jpg / .tif → image array
```

For DICOM files, the pixel array is extracted and converted into a normalized 2D image representation.

### 17.2 Medical Intensity Normalization

For CT images, the pixel values are windowed according to the selected medical mode.

Example:

```text
ct_lung → lung window
ct_soft → soft tissue window
ct_brain → brain window
```

For X-ray, MRI, and ultrasound, percentile normalization is often more suitable.

### 17.3 Embedding Extraction

The preprocessed image is passed into a frozen encoder.

```text
preprocessed image → frozen medical encoder → embedding vector
```

### 17.4 L2 Normalization

The embedding vector is normalized:

```text
embedding = embedding / ||embedding||
```

This allows dot product retrieval to behave like cosine similarity when vectors are normalized.

### 17.5 Memory Bank

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

### 17.6 Retrieval

During inference, CRISP-Med searches for the most similar stored medical embeddings.

```text
query embedding → top-k nearest neighbors
```

### 17.7 Voting

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

## 18. Recommended Experimental Variants

| Variant | Encoder | Retriever | Voting | Use Case |
|---|---|---|---|---|
| CRISP-Med-ResNet50-Numpy | `medical_resnet50` | NumPy | Weighted | Small CT/X-ray dataset |
| CRISP-Med-ResNet50-FAISS | `medical_resnet50` | FAISS | Weighted | Larger CT dataset |
| CRISP-Med-DenseNet121-Numpy | `medical_densenet121` | NumPy | Weighted | X-ray baseline |
| CRISP-Med-DenseNet121-FAISS | `medical_densenet121` | FAISS | Weighted | Larger X-ray dataset |
| CRISP-Med-EfficientNet-Numpy | `medical_efficientnet_b0` | NumPy | Weighted | Lightweight experiment |
| CRISP-Med-Herding | Any medical encoder | FAISS | Weighted | Memory-limited setting |
| CRISP-Med-Unknown | Any medical encoder | Any retriever | Weighted + Threshold | Unknown-class detection |

---

## 19. Suggested Evaluation Metrics

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

## 20. Recommended Dataset Splitting

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

## 21. Project Structure

```text
CRISP-Med/
├── pyproject.toml
├── README.md
├── LICENSE
├── docs/
│   ├── architecture.md
│   └── medical_usage.md
├── examples/
│   ├── medical_ctscan_usage.py
│   ├── medical_xray_usage.py
│   └── compare_medical_variants.py
├── tests/
│   ├── test_memory.py
│   ├── test_voting.py
│   └── test_medical_preprocessing.py
└── src/
    └── crisp/
        ├── __init__.py
        ├── classifier.py
        ├── cli.py
        ├── memory.py
        ├── prototypes.py
        ├── utils.py
        ├── voting.py
        ├── encoders/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── factory.py
        │   ├── resnet.py
        │   ├── clip_encoder.py
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

## 22. API Reference

### 22.1 `CRISPClassifier`

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

### 22.2 Medical `encoder_kwargs`

| Argument | Description | Example |
|---|---|---|
| `image_size` | Input image size | `224` |
| `intensity_mode` | Medical preprocessing mode | `ct_lung` |
| `normalize` | Normalization mode | `imagenet` |
| `checkpoint_path` | Optional pretrained medical checkpoint | `checkpoints/model.pth` |

### 22.3 Main Methods

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

## 23. Output Format

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
    "retriever": "faiss"
}
```

---

## 24. Notes and Limitations

- CRISP-Med is intended for research and education, not direct clinical diagnosis.
- The encoder is frozen during incremental updates.
- If the encoder or preprocessing mode is changed, the memory bank should be rebuilt.
- Threshold values must be tuned for each dataset and modality.
- DICOM images may have modality-specific metadata and intensity scaling; verify preprocessing before evaluation.
- For 3D CT/MRI, CRISP-Med currently operates on 2D slices unless a 3D encoder is added.
- Slice-level splitting can cause data leakage. Use patient-level splitting whenever possible.
- ImageNet-pretrained encoders are only baselines; better performance usually requires medical-domain pretraining or fine-tuning before freezing.
- Retrieval memory may grow large if all slices are stored. Use herding, prototype selection, or patient-level representative slice selection.
- Medical datasets are often imbalanced; report sensitivity, specificity, F1-score, and ROC-AUC in addition to accuracy.

---

## 25. Roadmap

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

---

## 26. License

This project is released under the MIT License.

---

## 27. Citation

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

## 28. Disclaimer

CRISP-Med does not provide medical advice, diagnosis, or treatment recommendations.  
All outputs should be interpreted only as machine learning experiment results and must not be used as a replacement for professional medical judgment.
