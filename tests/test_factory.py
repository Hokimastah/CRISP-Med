from crisp.encoders.factory import RESNET_NAMES, MEDICAL_NAMES


def test_encoder_name_sets():
    assert "resnet50" in RESNET_NAMES
    assert "medical_resnet50" in MEDICAL_NAMES
