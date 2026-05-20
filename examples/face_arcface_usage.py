from crisp import CRISPClassifier

clf = CRISPClassifier(
    encoder="arcface",
    retriever="numpy",
    top_k=5,
    voting="weighted",
    encoder_kwargs={
        "model_name": "buffalo_l",
        "det_size": (640, 640),
        "face_selection": "largest",
    },
)

clf.add_folder("faces_dataset")
clf.save("face_memory.pkl")

result = clf.predict("test_face.jpg", threshold=0.45)
print(result)
