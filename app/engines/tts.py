"""Kokoro-82M text-to-speech engine (ONNX, CPU).

Uses `onnx-community/Kokoro-82M-v1.0-ONNX` (`model_q8f16.onnx`, ~86 MB) with
per-voice style tensors (`voices/<name>.bin`).

English G2P is **self-contained**: `phonemizer-fork` + espeak-ng (shipped in the
`espeakng-loader` wheel). It deliberately does *not* depend on `misaki`, whose
releases require Python <3.13 (Botkeep runs 3.13) and whose `[en]` extra drags
in spacy + torch (~5.9 GB). The espeak→Kokoro phoneme mapping is ported from
misaki's `EspeakFallback` (MIT, https://github.com/hexgrad/misaki) so the phoneme
inventory matches what Kokoro was trained on.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.errors import ModelNotReadyError, SynthesisError

logger = logging.getLogger("apsides.voice.tts")

SAMPLE_RATE = 24000
MAX_TOKENS = 510          # model context is 512 incl. pad tokens
_CHUNK_CHAR_BUDGET = 300  # conservative split budget so phonemes stay < MAX_TOKENS


class EspeakG2P:
    """English grapheme-to-phoneme via espeak-ng, mapped to Kokoro phonemes.

    No misaki / spacy / torch. The ``_E2M`` table and post-processing mirror
    misaki's EspeakFallback (MIT) so the output matches Kokoro's training data.
    """

    # espeak IPA -> Kokoro IPA substitutions (ported from misaki, MIT)
    _E2M = sorted(
        {
            "\u0294\u02ccn\u0329": "\u0294n", "\u0294n\u0329": "\u0294n",
            "a^\u026a": "I", "a^\u028a": "W",
            "d^\u0292": "\u02a4",
            "e^\u026a": "A", "e": "A",
            "t^\u0283": "\u02a7",
            "\u0254^\u026a": "Y",
            "\u0259^l": "\u1d4al",
            "\u02b2o": "jo", "\u02b2\u0259": "j\u0259", "\u02b2": "",
            "\u025a": "\u0259\u0279",
            "r": "\u0279",
            "x": "k", "\u00e7": "k",
            "\u0250": "\u0259",
            "\u026c": "l",
            "\u0303": "",
        }.items(),
        key=lambda kv: -len(kv[0]),
    )

    def __init__(self, british: bool) -> None:
        import espeakng_loader
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        # Point phonemizer at the espeak-ng library/data bundled by the wheel.
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

        self._british = british
        self._backend = EspeakBackend(
            language=f"en-{'gb' if british else 'us'}",
            preserve_punctuation=True,
            with_stress=True,
            tie="^",
        )

    def __call__(self, text: str):
        result = self._backend.phonemize([text])
        if not result:
            return "", None
        ps = result[0].strip()
        for old, new in type(self)._E2M:
            ps = ps.replace(old, new)
        ps = re.sub(r"(\S)\u0329", "\u1d4a\\1", ps).replace(chr(809), "")
        if self._british:
            ps = ps.replace("e^\u0259", "\u025b\u02d0")
            ps = ps.replace("i\u0259", "\u026a\u0259")
            ps = ps.replace("\u0259^\u028a", "Q")
        else:
            ps = ps.replace("o^\u028a", "O")
            ps = ps.replace("\u025c\u02d0\u0279", "\u025c\u0279")
            ps = ps.replace("\u025c\u02d0", "\u025c\u0279")
            ps = ps.replace("\u026a\u0259", "i\u0259")
            ps = ps.replace("\u02d0", "")
        ps = ps.replace("o", "\u0254")
        ps = ps.replace("\u027e", "T").replace("\u0294", "t")
        return ps.replace("^", ""), None


class KokoroTTS:
    def __init__(
        self,
        model_path: str,
        voices_dir: str,
        vocab_path: str,
        num_threads: int = 1,
        default_voice: str = "af_heart",
    ) -> None:
        self.model_path = Path(model_path)
        self.voices_dir = Path(voices_dir)
        self.vocab_path = Path(vocab_path)
        self.num_threads = max(1, int(num_threads))
        self.default_voice = default_voice

        self._session = None
        self._vocab: Dict[str, int] = {}
        self._voice_cache: Dict[str, np.ndarray] = {}
        self._g2p_cache: Dict[str, object] = {}
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------
    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"TTS model not found: {self.model_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"TTS vocab not found: {self.vocab_path}")

        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Trim peak memory so the ONNX session coexists with the STT runtime on
        # 2 GB hosts (the memory arena can otherwise reserve a large block).
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        with open(self.vocab_path, "r", encoding="utf-8") as fh:
            vocab = json.load(fh)
        self._vocab = vocab.get("vocab", vocab)
        logger.info("Kokoro TTS loaded (%d vocab entries)", len(self._vocab))

    @property
    def ready(self) -> bool:
        return self._session is not None and bool(self._vocab)

    # -- voices ---------------------------------------------------------
    def list_voices(self) -> List[str]:
        if not self.voices_dir.exists():
            return []
        voices = []
        for p in sorted(self.voices_dir.glob("*.bin")):
            stem = p.stem
            # named voices look like af_heart / bm_george; skip bare 'af.bin'
            if len(stem) > 2 and stem[2] == "_":
                voices.append(stem)
        return voices

    def _voice_file(self, voice: str) -> Path:
        return self.voices_dir / f"{voice}.bin"

    def _load_voice(self, voice: str) -> np.ndarray:
        if voice in self._voice_cache:
            return self._voice_cache[voice]
        path = self._voice_file(voice)
        if not path.exists():
            raise SynthesisError(f"Unknown voice '{voice}'.")
        arr = np.fromfile(path, dtype=np.float32).reshape(-1, 256)
        self._voice_cache[voice] = arr
        return arr

    def _style(self, voice: str, n_tokens: int) -> np.ndarray:
        arr = self._load_voice(voice)
        idx = min(max(n_tokens, 0), arr.shape[0] - 1)
        return arr[idx: idx + 1]  # (1, 256)

    # -- text front-end -------------------------------------------------
    def _g2p(self, voice: str):
        # voice[0]: 'a' = American English, 'b' = British English
        accent = "b" if voice.startswith("b") else "a"
        if accent in self._g2p_cache:
            return self._g2p_cache[accent]
        british = accent == "b"
        try:
            g2p = EspeakG2P(british)
        except Exception as exc:  # noqa: BLE001
            raise ModelNotReadyError(
                "English G2P is unavailable. Install `phonemizer-fork` and "
                "`espeakng-loader`."
            ) from exc
        self._g2p_cache[accent] = g2p
        return g2p

    @staticmethod
    def _split_text(text: str, budget: int = _CHUNK_CHAR_BUDGET) -> List[str]:
        text = " ".join(text.split())
        if not text:
            return []
        parts = re.split(r"(?<=[.!?;:])\s+", text)
        chunks: List[str] = []
        cur = ""
        for part in parts:
            if not cur:
                cur = part
            elif len(cur) + 1 + len(part) <= budget:
                cur = f"{cur} {part}"
            else:
                chunks.append(cur)
                cur = part
            while len(cur) > budget:  # hard-split over-long runs
                chunks.append(cur[:budget])
                cur = cur[budget:]
        if cur:
            chunks.append(cur)
        return [c for c in chunks if c.strip()]

    # -- synthesis ------------------------------------------------------
    def synthesize(self, text: str, voice: Optional[str], speed: float) -> np.ndarray:
        if not self.ready:
            raise ModelNotReadyError("TTS model is not loaded.")
        voice = voice or self.default_voice
        g2p = self._g2p(voice)

        pieces: List[np.ndarray] = []
        with self._lock:  # onnxruntime run is thread-safe, but serialise for CPU budget
            for chunk in self._split_text(text):
                try:
                    phonemes, _ = g2p(chunk)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("G2P failed on a chunk: %s", exc)
                    continue
                ids = [self._vocab[c] for c in phonemes if c in self._vocab]
                if not ids:
                    continue
                ids = ids[:MAX_TOKENS]
                input_ids = np.array([[0, *ids, 0]], dtype=np.int64)
                style = self._style(voice, len(ids))
                out = self._session.run(
                    None,
                    {
                        "input_ids": input_ids,
                        "style": style,
                        "speed": np.array([float(speed)], dtype=np.float32),
                    },
                )[0]
                pieces.append(np.asarray(out).reshape(-1))

        if not pieces:
            raise SynthesisError("No synthesizable content in the given text.")
        audio = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        return audio.astype(np.float32)
