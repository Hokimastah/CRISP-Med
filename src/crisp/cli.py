from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import torch

from .classifier import CRISPClassifier
from .trainer import train_medical_backbone


def parse_json(value: Optional[str]) -> Dict[str, Any]:
    if value is None or value == "":
        return {}

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {value}") from exc


def merge_checkpoint_kwargs(
    encoder_kwargs: Dict[str, Any],
    checkpoint: Optional[str],
) -> Dict[str, Any]:
    kwargs = dict(encoder_kwargs or {})

    if checkpoint:
        kwargs["checkpoint_path"] = checkpoint

        try:
            ckpt = torch.load(checkpoint, map_location="cpu")

            if isinstance(ckpt, dict):
                preprocessing = ckpt.get("preprocessing", {})

                for key, value in preprocessing.items():
                    kwargs.setdefault(key, value)
        except Exception:
            pass

    return kwargs


def command_train(args) -> None:
    train_medical_backbone(
        data_dir=args.data,
        encoder=args.encoder,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        intensity_mode=args.intensity_mode,
        normalize=args.normalize,
        val_split=args.val_split,
        pretrained=not args.no_pretrained,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        class_weighted_loss=args.class_weighted_loss,
    )


def command_index(args) -> None:
    encoder_kwargs = parse_json(args.encoder_kwargs)
    retriever_kwargs = parse_json(args.retriever_kwargs)

    encoder_kwargs = merge_checkpoint_kwargs(
        encoder_kwargs=encoder_kwargs,
        checkpoint=args.checkpoint,
    )

    clf = CRISPClassifier(
        encoder=args.encoder,
        retriever=args.retriever,
        pretrained=not args.no_pretrained,
        device=args.device,
        top_k=args.top_k,
        voting=args.voting,
        encoder_kwargs=encoder_kwargs,
        retriever_kwargs=retriever_kwargs,
    )

    clf.add_folder(args.data)
    clf.save(args.output)

    print(f"Indexed samples: {len(clf)}")
    print(f"Saved memory: {args.output}")


def command_predict(args) -> None:
    encoder_kwargs = parse_json(args.encoder_kwargs)
    retriever_kwargs = parse_json(args.retriever_kwargs)

    encoder_kwargs = merge_checkpoint_kwargs(
        encoder_kwargs=encoder_kwargs,
        checkpoint=args.checkpoint,
    )

    clf = CRISPClassifier(
        encoder=args.encoder,
        retriever=args.retriever,
        pretrained=not args.no_pretrained,
        device=args.device,
        top_k=args.top_k,
        voting=args.voting,
        encoder_kwargs=encoder_kwargs,
        retriever_kwargs=retriever_kwargs,
    )

    clf.load(args.memory)

    result = clf.predict(
        image_path=args.image,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crisp",
        description="CRISP-Med command line interface",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train or fine-tune medical backbone before indexing.",
    )

    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--encoder", default="medical_resnet50")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--intensity-mode", default="percentile")
    train_parser.add_argument("--normalize", default="imagenet")
    train_parser.add_argument("--val-split", type=float, default=0.2)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--class-weighted-loss", action="store_true")
    train_parser.set_defaults(func=command_train)

    index_parser = subparsers.add_parser(
        "index",
        help="Build memory bank from folder dataset.",
    )

    index_parser.add_argument("--data", required=True)
    index_parser.add_argument("--output", required=True)
    index_parser.add_argument("--encoder", default="medical_resnet50")
    index_parser.add_argument("--retriever", default="numpy")
    index_parser.add_argument("--checkpoint", default=None)
    index_parser.add_argument("--top-k", type=int, default=5)
    index_parser.add_argument("--voting", default="weighted", choices=["weighted", "majority"])
    index_parser.add_argument("--device", default=None)
    index_parser.add_argument("--no-pretrained", action="store_true")
    index_parser.add_argument("--encoder-kwargs", default="{}")
    index_parser.add_argument("--retriever-kwargs", default="{}")
    index_parser.set_defaults(func=command_index)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict single image using saved memory bank.",
    )

    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--memory", required=True)
    predict_parser.add_argument("--encoder", default="medical_resnet50")
    predict_parser.add_argument("--retriever", default="numpy")
    predict_parser.add_argument("--checkpoint", default=None)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--threshold", type=float, default=None)
    predict_parser.add_argument("--voting", default="weighted", choices=["weighted", "majority"])
    predict_parser.add_argument("--device", default=None)
    predict_parser.add_argument("--no-pretrained", action="store_true")
    predict_parser.add_argument("--encoder-kwargs", default="{}")
    predict_parser.add_argument("--retriever-kwargs", default="{}")
    predict_parser.set_defaults(func=command_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()