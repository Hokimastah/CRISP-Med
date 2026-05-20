from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="resnet50",
    retriever="numpy",
    top_k=5,
    voting="weighted",
)

clf.add_folder("dataset")
clf.save("memory_bank.pkl")

result = clf.predict("test_image.jpg")
print(result)
