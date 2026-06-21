"""
push_to_hub.py — publish the model + checkpoints for BabyLM submission
======================================================================

The BabyLM pipeline loads your model from the HuggingFace Hub and expects:
  * the FINAL model on the `main` branch (train_babylm.py writes the BEST-BLiMP
    checkpoint here, not the end-of-training weights)
  * each training snapshot on its OWN branch, named exactly like the local dir
    (chck_1M, chck_2M, ..., chck_10M, chck_20M, ...) so the `--fast` checkpoint
    eval can find them.

Run on Kaggle AFTER train_babylm.py, with a write token:
    huggingface-cli login            # or set HF_TOKEN env var
    python push_to_hub.py --repo_id YOUR_USERNAME/conv-induction-babylm-strict-small

The repo MUST be public before you submit. Because the architecture is custom, anyone
loading it (including the eval pipeline) must pass trust_remote_code=True — the
modeling_induction.py file is uploaded with each revision so that works.

NOTE: save_pretrained only writes the auto_map entry for the class it saved
(AutoModelForCausalLM). The GLUE fine-tuning loads the model via AutoModel and reads
config.hidden_size, so before uploading the final model we rewrite final-model/config.json
to expose ALL auto classes + hidden_size. (Checkpoint branches are zero-shot CausalLM only,
so they don't need this.)
"""

import os, re, json, argparse
from huggingface_hub import HfApi, create_repo, create_branch

OUT = "/kaggle/working"
CKPT_RE = re.compile(r"^chck_(\d+)M$")

FULL_AUTO_MAP = {
    "AutoConfig": "modeling_induction.InductionConfig",
    "AutoModel": "modeling_induction.InductionModel",
    "AutoModelForCausalLM": "modeling_induction.InductionForCausalLM",
    "AutoModelForSequenceClassification": "modeling_induction.InductionForSequenceClassification",
}


def patch_final_config(final_dir):
    """Ensure the final model exposes every auto class + hidden_size for the GLUE loader."""
    cfg_path = os.path.join(final_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.setdefault("auto_map", {})
    cfg["auto_map"].update(FULL_AUTO_MAP)
    cfg["hidden_size"] = cfg.get("hidden_size", cfg.get("d_model", 384))
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("  patched final-model/config.json: auto_map (+AutoModel/SeqClf) + hidden_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, help="e.g. yourname/conv-induction-babylm-strict-small")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--private", action="store_true", help="create private (make public before submitting!)")
    ap.add_argument("--skip_checkpoints", action="store_true", help="push only the final model to main")
    args = ap.parse_args()

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # 1) final model (best-BLiMP checkpoint) -> main
    final = os.path.join(args.out, "final-model")
    patch_final_config(final)                                   # <-- GLUE fix
    print(f"Uploading final model ({final}) -> main")
    api.upload_folder(folder_path=final, repo_id=args.repo_id, commit_message="final model (best-BLiMP)")

    # 1b) model card -> main (if a README.md sits next to the script or in out/)
    for cand in ("README.md", os.path.join(args.out, "README.md")):
        if os.path.exists(cand):
            print(f"Uploading model card ({cand}) -> main")
            api.upload_file(path_or_fileobj=cand, path_in_repo="README.md",
                            repo_id=args.repo_id, commit_message="model card")
            break

    # 2) each milestone snapshot -> its own branch (learning-curve / --fast eval)
    if not args.skip_checkpoints:
        ckpt_root = os.path.join(args.out, "checkpoints")
        names = [n for n in os.listdir(ckpt_root)
                 if CKPT_RE.match(n) and os.path.isdir(os.path.join(ckpt_root, n))]
        for name in sorted(names, key=lambda s: int(CKPT_RE.match(s).group(1))):
            path = os.path.join(ckpt_root, name)
            print(f"Uploading {name} -> branch '{name}'")
            create_branch(args.repo_id, branch=name, exist_ok=True)
            api.upload_folder(folder_path=path, repo_id=args.repo_id, revision=name,
                              commit_message=f"checkpoint {name}")

    print(f"\nDone. https://huggingface.co/{args.repo_id}")
    print("Remember: the repo must be PUBLIC before submitting to the leaderboard.")


if __name__ == "__main__":
    main()
