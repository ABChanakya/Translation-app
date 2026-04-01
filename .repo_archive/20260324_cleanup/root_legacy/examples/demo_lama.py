# ──────────────── demo_lama.py ────────────────
# this is for re drawing the image after puttin the new image and mask
import os
import subprocess
import argparse

def run_command(cmd):
    print(f"▶︎ {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"Command failed (exit {ret}):\n  {cmd}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal LaMa inpainting demo: "
                    "downloads a pretrained model, "
                    "prepares one image+mask pair, "
                    "and runs bin/predict.py."
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to a single RGB image (e.g. input.png) to inpaint"
    )
    parser.add_argument(
        "--mask", type=str, required=True,
        help="Path to a single grayscale mask (white=hole) (e.g. input_mask.png)"
    )
    parser.add_argument(
        "--outdir", type=str, default="lama_output",
        help="Directory where inpainted result will be written"
    )
    parser.add_argument(
        "--model-dir", type=str, default="big_lama_model",
        help="Directory in which to download/unzip the pretrained model"
    )
    args = parser.parse_args()

    # 1) Clone the LaMa repo (if not already present)
    if not os.path.isdir("lama"):
        run_command("git clone https://github.com/advimman/lama.git")

    # 2) Download a “big-lama” pretrained checkpoint from HuggingFace
    #    (Places2 “Big LaMa” model ~ 500 MB). Unzip into args.model_dir.
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)
        # Change to the model directory to download
        run_command(f"curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip")
        run_command("unzip big-lama.zip -d big_lama_model")
        run_command("rm big-lama.zip")

    # 3) Create a temporary folder with our single image+mask pair.
    demo_folder = "demo_pair"
    if not os.path.isdir(demo_folder):
        os.makedirs(demo_folder)
    # Copy the user-specified image and mask into demo_folder.
    # Name them so that LaMa’s default suffix logic picks them up:
    #   ───>  <basename>.png    and  <basename>_mask.png
    import shutil
    base_name = "image1"
    img_dest  = os.path.join(demo_folder, f"{base_name}.png")
    msk_dest  = os.path.join(demo_folder, f"{base_name}_mask.png")
    shutil.copy(args.image, img_dest)
    shutil.copy(args.mask, msk_dest)

    # 4) Prepare output directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    # 5) Set environment variables so that Python sees “lama/” as a module:
    os.environ["TORCH_HOME"] = os.path.abspath(args.model_dir)
    os.environ["PYTHONPATH"] = os.path.abspath("lama")

    # 6) Call LaMa’s built-in prediction script on our folder:
    #    We need to tell predict.py:
    #      – model.path=…    → location of “big_lama_model/models/best.ckpt”
    #      – indir=…         → folder with image1.png & image1_mask.png
    #      – outdir=…        → where to write image1_inpainted.png
    #
    #    (If you want refinement, append “refine=True”.)
    #
    cmd = (
        "python3 lama/bin/predict.py "
        f"model.path={os.path.abspath(args.model_dir)}/models/best.ckpt "
        f"indir={os.path.abspath(demo_folder)} "
        f"outdir={os.path.abspath(args.outdir)}"
    )
    run_command(cmd)

    print("\n✅  Finished. Your inpainted image is here:")
    print(f"      {os.path.abspath(args.outdir)}/{base_name}.png")
    print("   └→ (LaMa will overwrite the original “.png”; examine outdir to see the new image.)")
