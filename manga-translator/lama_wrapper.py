# lama_wrapper.py  – minimal, safe wrapper around LaMa CLI
import os, subprocess, tempfile, shutil, uuid, glob

_MODEL_DIR = "big_lama_model"        # ← the folder that will hold config.yaml + models/
#_CHECKPOINT_NAME = "best.ckpt"       # ← what we want LaMa to load
_CHECKPOINT_NAME = "big-lama.safetensors"   # <- instead of "best.ckpt"


# --------------------------------------------------------------------------- #
def _run(cmd: str) -> None:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")

# --------------------------------------------------------------------------- #
def _ensure_model() -> None:
    """
    Clone LaMa (if missing) and download the 500 MB ‘big-lama’ weights once.
    """
    if _find_ckpt():                       # we already have best.ckpt somewhere
        return

    if not os.path.isdir("lama"):
        _run("git clone https://github.com/advimman/lama.git")

    if not os.path.exists("big-lama.zip"):
        _run("curl -L -o big-lama.zip "
             "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip")

    _run("unzip -o big-lama.zip -d big_lama_model")
    # the zip contains a redundant “big-lama/” sub-folder – flatten it
    _run("rsync -a --remove-source-files big_lama_model/big-lama/ big_lama_model/")
    _run("rm -rf big_lama_model/big-lama")

# --------------------------------------------------------------------------- #
def _find_ckpt() -> str | None:
    hits = glob.glob(os.path.join(_MODEL_DIR, "**", _CHECKPOINT_NAME), recursive=True)
    return os.path.abspath(hits[0]) if hits else None

# --------------------------------------------------------------------------- #
def inpaint(pil_image, pil_mask):
    """
    Simple convenience wrapper around  `python -m lama.bin.predict …`
    Returns: PIL.Image with the masked area in-painted.
    """
    _ensure_model()
    ckpt_path = _find_ckpt()                              # …/big_lama_model/models/best.ckpt
    model_dir = os.path.dirname(os.path.dirname(ckpt_path))  # …/big_lama_model

    work = tempfile.mkdtemp(prefix="lama_")
    img = os.path.join(work, "img.png")
    msk = os.path.join(work, "img_mask.png")
    pil_image.save(img)
    pil_mask.save(msk)

    # Make LaMa’s Python sources importable for the subprocess
    #os.environ["PYTHONPATH"] = os.path.abspath("lama")
    os.environ["PYTHONPATH"] = (
        os.path.abspath(".") + os.pathsep + os.path.abspath("lama") 
    )


    _run(
        f"python -m lama.bin.predict "
        f"model.path={model_dir} "
        f"model.checkpoint={_CHECKPOINT_NAME} "
        f"indir={work} outdir={work} device=cpu"
    )

    from PIL import Image
    out_img = Image.open(img).convert("RGB")
    shutil.rmtree(work, ignore_errors=True)
    return out_img
