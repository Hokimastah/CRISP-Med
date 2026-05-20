from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="medical_densenet121",
    retriever="numpy",
    top_k=5,
    voting="weighted",
    encoder_kwargs={
        "intensity_mode": "percentile",
        "image_size": 224,
    },
)

clf.add_folder("xray_dataset")
clf.save("xray_memory.pkl")

result = clf.predict("test_xray.png", threshold=0.50)
print(result)
