# CRISP Architecture

CRISP v1 is a retrieval-based incremental image classifier.

```text
Input Image
  ↓
Frozen Encoder
  ↓
L2-normalized Embedding
  ↓
Memory Bank
  ↓
Retriever Index
  ↓
Top-k Neighbors
  ↓
Weighted / Majority Voting
  ↓
Predicted Label
```

## Core components

- `CRISPClassifier`: orchestration class.
- `encoders`: feature extractors.
- `MemoryBank`: stores embeddings, labels, and metadata.
- `retrievers`: vector search backends.
- `voting`: converts top-k neighbors into class scores.
- `prototypes`: optional memory compression.

## ArcFace path

```text
Face image
→ InsightFace detection and alignment
→ ArcFace 512-d embedding
→ MemoryBank
→ Top-k retrieval
→ Identity prediction
```

## Medical image path

```text
CT/X-ray/DICOM image
→ grayscale or DICOM pixel reading
→ intensity normalization / CT windowing
→ RGB conversion for 2D backbone
→ medical_* encoder
→ MemoryBank
→ Top-k retrieval
→ class prediction
```
