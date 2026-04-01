# Manga Colorization

This folder contains the bundled manga colorization code and weights used by the public `Colorize` page.

## Current Role In The Project

Colorization is intentionally kept as a secondary public feature of Auto Manga Translation.

Relevant runtime code:

- public integration: [../src/colorization_service.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/colorization_service.py)
- web route: [../web/app.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/web/app.py)

The UI only exposes the feature when the required assets are available. If weights are missing, the public page shows a setup-needed state instead of failing silently.

## How To Run Colorization Directly

Install the extra requirements in this folder and place the expected weights where the colorization code expects them.

Typical direct usage:

```bash
python inference.py -p "path/to/file-or-folder"
```

## Expected Assets

The bundled colorization code expects generator/extractor weights under `networks/` and denoiser weights under `denoising/models/`.

If those files are missing, the public UI will not claim the feature is ready.

## Folder Contents

Key files:

- [inference.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/colorization/inference.py)
- [colorizator.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/colorization/colorizator.py)
- [requirements.txt](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/colorization/requirements.txt)

## LaMa / Inpainting Note

Older documentation in this folder separately explained LaMa-based text removal. That explanation is now folded into this single remaining guide.

Important distinction:

- `Colorization` adds color to black-and-white manga pages
- `Inpainting` removes source text during translation

The current inpainting path used by the translation pipeline is:

- [../src/models/inpainter.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/models/inpainter.py)
- [../src/pipeline.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/pipeline.py)

That inpainting layer currently talks to a LaMa service when available and falls back safely when it is not.

## If Colorization Is Not Working

Check:

1. the required weights exist in the expected subfolders
2. the extra dependencies from [requirements.txt](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/colorization/requirements.txt) are installed
3. [../src/colorization_service.py](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/src/colorization_service.py) reports the feature as available

## Related Docs

- Main project overview: [../README.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/README.md)
- Quickstart: [../QUICKSTART.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/QUICKSTART.md)
- Larger engineering guide: [../DEVELOPER_GUIDE.md](/home/chanakya/chanakya/Translation_tool-2/manga_translator_clean/DEVELOPER_GUIDE.md)
