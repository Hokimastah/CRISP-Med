from __future__ import annotations

import argparse
import json
from pprint import pprint
from typing import Any, Dict, Optional

from .classifier import CRISPClassifier


def _json_dict(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Value must be a JSON object/dict.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crisp",
        description="CRISP: Continual Retrieval & Indexing System for Perception",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--encoder", default="resnet50")
    common.add_argument("--retriever", default="numpy", choices=["numpy", "annoy", "faiss"])
    common.add_argument("--device", default=None)
    common.add_argument("--top-k", type=int, default=5)
    common.add_argument("--voting", default="weighted", choices=["weighted", "majority"])
    common.add_argument("--no-pretrained", action="store_true")
    common.add_argument("--encoder-kwargs", type=_json_dict, default={})
    common.add_argument("--retriever-kwargs", type=_json_dict, default={})

    index_parser = subparsers.add_parser("index", parents=[common], help="Index a folder dataset")
    index_parser.add_argument("--data", required=True, help="Folder dataset path")
    index_parser.add_argument("--output", required=True, help="Output memory .pkl path")
    index_parser.add_argument("--max-per-class", type=int, default=None, help="Optional herding compression limit per class")

    predict_parser = subparsers.add_parser("predict", parents=[common], help="Predict a single image")
    predict_parser.add_argument("--image", required=True, help="Image path")
    predict_parser.add_argument("--memory", required=True, help="Memory .pkl path")
    predict_parser.add_argument("--threshold", type=float, default=None, help="Optional unknown threshold")

    return parser


def _make_classifier(args: argparse.Namespace) -> CRISPClassifier:
    return CRISPClassifier(
        encoder=args.encoder,
        retriever=args.retriever,
        pretrained=not args.no_pretrained,
        device=args.device,
        top_k=args.top_k,
        voting=args.voting,
        encoder_kwargs=args.encoder_kwargs,
        retriever_kwargs=args.retriever_kwargs,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        clf = _make_classifier(args)
        clf.add_folder(args.data)
        if args.max_per_class is not None:
            clf.compress_memory(args.max_per_class)
        clf.save(args.output)
        print(f"Indexed {len(clf)} embeddings into: {args.output}")
        return

    if args.command == "predict":
        clf = _make_classifier(args)
        clf.load(args.memory)
        result = clf.predict(args.image, top_k=args.top_k, threshold=args.threshold)
        pprint(result)
        return


if __name__ == "__main__":
    main()
