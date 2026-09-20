"""Compare two related keyword sets with the real MiniLM model."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuego.embedding import DIMENSIONS, MODEL_NAME, Embedder, EmbeddingError

KEY_A = ["solar energy", "wind power", "renewable electricity", "carbon reduction", "climate protection",
         "clean technology", "battery storage", "electric vehicles", "energy efficiency", "sustainable development",
         "green investment", "power grid", "fossil fuels", "emissions targets", "decarbonization",
         "photovoltaic panels", "wind turbines", "energy transition", "environmental policy", "net zero"]
KEY_B = ["sunlight generation", "wind-generated electricity", "renewable power", "cutting carbon pollution", "climate action",
         "low-carbon innovation", "stored electricity", "battery-powered cars", "energy conservation", "sustainable growth",
         "climate finance", "electricity network", "coal and oil", "pollution limits", "carbon elimination",
         "solar cells", "wind generators", "clean power shift", "green regulation", "carbon neutrality"]


def compare_vectors(a, b):
    if len(a) != DIMENSIONS or len(b) != DIMENSIONS:
        raise ValueError("Expected two 384-dimensional vectors.")
    if not all(math.isfinite(x) for x in [*a, *b]):
        raise ValueError("Vector values must be finite.")
    norm_a, norm_b = math.hypot(*a), math.hypot(*b)
    if not math.isclose(norm_a, 1, abs_tol=1e-6) or not math.isclose(norm_b, 1, abs_tol=1e-6):
        raise ValueError("Both vectors must have unit length.")
    similarity = max(-1.0, min(1.0, math.fsum(x*y for x,y in zip(a,b))/(norm_a*norm_b)))
    return {"norm_a":norm_a, "norm_b":norm_b, "cosine_similarity":similarity,
            "cosine_distance":1-similarity, "euclidean_distance":math.dist(a,b)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Embed two related sets of 20 different terms and compare them.")
    parser.add_argument("--min-similarity",type=float,default=0.7,help="Sample smoke threshold. Default: 0.7; not a universal cutoff.")
    parser.add_argument("--device",choices=("cpu",),default="cpu")
    parser.add_argument("--cache-folder",help="Model cache directory.")
    parser.add_argument("--local-files-only",action="store_true",help="Use the model cache without downloading.")
    args=parser.parse_args(argv)
    if not math.isfinite(args.min_similarity) or not -1 <= args.min_similarity <= 1:
        parser.error("--min-similarity must be between -1 and 1.")
    print("Loading real model: "+MODEL_NAME, file=sys.stderr,flush=True)
    print("First use may download model files. No NVIDIA key is needed.",file=sys.stderr,flush=True)
    start=time.monotonic()
    try:
        embedder=Embedder(device=args.device,cache_folder=args.cache_folder,local_files_only=args.local_files_only)
        vectors=embedder.embed_many(["; ".join(KEY_A),"; ".join(KEY_B)])
        metrics=compare_vectors(*vectors)
    except (EmbeddingError,ValueError,OSError) as exc:
        print("FAIL [embedding]: "+str(exc),file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("CANCELLED",file=sys.stderr)
        return 130
    passed=metrics["cosine_similarity"]>=args.min_similarity
    print(json.dumps({"model":MODEL_NAME,"backend":"fastembed-onnxruntime","dimensions":DIMENSIONS,
                      "key_a":{"keywords":KEY_A,"vector":vectors[0]},
                      "key_b":{"keywords":KEY_B,"vector":vectors[1]},
                      "metrics":metrics,"min_similarity":args.min_similarity,"passed":passed},indent=2,allow_nan=False))
    print(f"{'PASS' if passed else 'FAIL'}: cosine similarity={metrics['cosine_similarity']:.6f}, "
          f"cosine distance={metrics['cosine_distance']:.6f}, Euclidean distance={metrics['euclidean_distance']:.6f}; "
          f"elapsed={time.monotonic()-start:.2f}s. Threshold is for this sample only.",file=sys.stderr,flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
