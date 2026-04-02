"""Translation engine registry with availability checks for public/admin UIs."""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class EngineSpec:
    """Declarative metadata for one translator engine."""

    engine_id: str
    label: str
    factory_name: str
    module_path: str | None = None
    dependency_modules: tuple[str, ...] = ()
    required_env_vars: tuple[str, ...] = ()
    public_visible: bool = True
    admin_visible: bool = True
    implemented: bool = True
    preferred_rank: int = 100
    enable_instructions: str = ""
    availability_check: Callable[[], tuple[bool, str | None]] | None = None


@dataclass(frozen=True)
class EngineStatus:
    """Resolved engine availability info for rendering and defaults."""

    engine_id: str
    label: str
    factory_name: str
    enabled: bool
    public_visible: bool
    admin_visible: bool
    implemented: bool
    required_env_vars: list[str] = field(default_factory=list)
    missing_env_vars: list[str] = field(default_factory=list)
    missing_dependencies: list[str] = field(default_factory=list)
    disable_reason: str | None = None
    enable_instructions: str = ""
    preferred_rank: int = 100


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _check_ollama_service() -> tuple[bool, str | None]:
    if not _module_available("ollama"):
        return False, "Missing Python dependency: ollama"
    try:
        ollama = importlib.import_module("ollama")
        ollama.list()
    except Exception as exc:
        return False, f"Ollama unavailable: {exc}"
    return True, None


ENGINE_SPECS: tuple[EngineSpec, ...] = (
    EngineSpec(
        engine_id="gemma3",
        label="Gemma 3 (Ollama)",
        factory_name="Gemma3",
        module_path="src.translators.gemma.GemmaTranslator",
        dependency_modules=("ollama",),
        preferred_rank=10,
        enable_instructions="Install Ollama, pull the configured Gemma model, and start the Ollama service.",
        availability_check=_check_ollama_service,
    ),
    EngineSpec(
        engine_id="translategemma",
        label="TranslateGemma (Ollama)",
        factory_name="TranslateGemma",
        module_path="src.translators.translategemma.TranslateGemmaTranslator",
        dependency_modules=("ollama",),
        preferred_rank=12,
        enable_instructions="Install Ollama, run 'ollama pull translategemma:12b', and start the Ollama service.",
        availability_check=_check_ollama_service,
    ),
    EngineSpec(
        engine_id="google",
        label="Google Translate",
        factory_name="Google",
        module_path="src.translators.google.GoogleTranslator",
        dependency_modules=("deep_translator",),
        preferred_rank=20,
        enable_instructions="Install the 'deep-translator' package.",
    ),
    EngineSpec(
        engine_id="deepl",
        label="DeepL API",
        factory_name="DeepL",
        module_path="src.translators.deepl.DeepLTranslator",
        dependency_modules=("deepl",),
        required_env_vars=("DEEPL_API_KEY",),
        preferred_rank=30,
        enable_instructions="Install the 'deepl' package and set DEEPL_API_KEY.",
    ),
    EngineSpec(
        engine_id="argos",
        label="Argos Translate",
        factory_name="Argos",
        module_path="src.translators.offline.OfflineTranslator",
        dependency_modules=("argostranslate", "requests"),
        preferred_rank=40,
        enable_instructions="Install argostranslate and allow the language pack download/cache step.",
    ),
    EngineSpec(
        engine_id="marianmt",
        label="MarianMT",
        factory_name="MarianMT",
        module_path="src.translators.offline.OfflineTranslator",
        dependency_modules=("transformers",),
        preferred_rank=50,
        enable_instructions="Install transformers and download the needed MarianMT model weights.",
    ),
    EngineSpec(
        engine_id="nllb",
        label="NLLB",
        factory_name="NLLB",
        module_path="src.translators.offline.OfflineTranslator",
        dependency_modules=("transformers",),
        preferred_rank=60,
        enable_instructions="Install transformers and download the configured NLLB model weights.",
    ),
    EngineSpec(
        engine_id="azure",
        label="Azure Translator",
        factory_name="Azure",
        dependency_modules=(),
        required_env_vars=("AZURE_TRANSLATOR_KEY", "AZURE_ENDPOINT"),
        public_visible=False,
        admin_visible=True,
        implemented=False,
        preferred_rank=70,
        enable_instructions="Add an Azure translator implementation, then set AZURE_TRANSLATOR_KEY and AZURE_ENDPOINT.",
    ),
)


def get_engine_spec(engine_id: str) -> EngineSpec:
    normalized = engine_id.strip().lower()
    for spec in ENGINE_SPECS:
        if spec.engine_id == normalized or spec.factory_name.lower() == normalized:
            return spec
    raise KeyError(f"Unknown engine: {engine_id}")


def get_engine_status(engine_id: str) -> EngineStatus:
    spec = get_engine_spec(engine_id)
    missing_dependencies = [name for name in spec.dependency_modules if not _module_available(name)]
    missing_env_vars = [name for name in spec.required_env_vars if not os.getenv(name)]

    enabled = spec.implemented and not missing_dependencies and not missing_env_vars
    disable_reason = None

    if not spec.implemented:
        enabled = False
        disable_reason = "Engine is registered for future support but not implemented yet."
    elif missing_dependencies:
        disable_reason = "Missing dependencies: " + ", ".join(missing_dependencies)
    elif missing_env_vars:
        disable_reason = "Missing environment variables: " + ", ".join(missing_env_vars)

    if enabled and spec.availability_check:
        enabled, availability_reason = spec.availability_check()
        if not enabled:
            disable_reason = availability_reason

    return EngineStatus(
        engine_id=spec.engine_id,
        label=spec.label,
        factory_name=spec.factory_name,
        enabled=enabled,
        public_visible=spec.public_visible,
        admin_visible=spec.admin_visible,
        implemented=spec.implemented,
        required_env_vars=list(spec.required_env_vars),
        missing_env_vars=missing_env_vars,
        missing_dependencies=missing_dependencies,
        disable_reason=disable_reason,
        enable_instructions=spec.enable_instructions,
        preferred_rank=spec.preferred_rank,
    )


def list_engine_statuses(scope: str = "public") -> list[EngineStatus]:
    statuses = [get_engine_status(spec.engine_id) for spec in ENGINE_SPECS]
    if scope == "public":
        statuses = [status for status in statuses if status.public_visible and status.enabled]
    elif scope == "admin":
        statuses = [status for status in statuses if status.admin_visible]
    return sorted(statuses, key=lambda item: (not item.enabled, item.preferred_rank, item.label.lower()))


def get_default_engine_status() -> EngineStatus | None:
    public_engines = list_engine_statuses(scope="public")
    return public_engines[0] if public_engines else None

