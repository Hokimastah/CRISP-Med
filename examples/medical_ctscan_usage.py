from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_resnet50",
    retriever="numpy",
    top_k=7,
    voting="weighted",
    encoder_kwargs={
        "intensity_mode": "ct_lung",
        "image_size": 224,
    },
)

clf.add_folder("dataset_ct")
clf.compress_memory(max_per_class=30)
clf.save("medical_memory.pkl")

result = clf.predict("test_slice.dcm", threshold=0.55)
print(result)
