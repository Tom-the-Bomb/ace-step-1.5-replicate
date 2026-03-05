import os
import subprocess
import tempfile
from pathlib import Path as PathLib
from typing import List, Optional

import logging

from cog import BasePredictor, Input, Path

from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.llm_inference import LLMHandler

log = logging.getLogger(__name__)

DIT_MODEL = "acestep-v15-turbo"
LM_MODEL = "acestep-5Hz-lm-4B"


class Predictor(BasePredictor):
    def setup(self) -> None:
        project_root = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = PathLib(project_root) / "weights"

        if not (checkpoint_dir / DIT_MODEL).exists():
            log.info("Downloading main model weights...")
            subprocess.run(
                ["acestep-download", "-d", str(checkpoint_dir)],
                check=True,
            )
        if not (checkpoint_dir / LM_MODEL).exists():
            log.info("Downloading %s weights...", LM_MODEL)
            subprocess.run(
                ["acestep-download", "-d", str(checkpoint_dir), "-m", LM_MODEL],
                check=True,
            )

        log.info("Loading DiT model: %s", DIT_MODEL)
        self.dit_handler = AceStepHandler()
        init_status, _ = self.dit_handler.initialize_service(
            project_root=project_root,
            config_path=DIT_MODEL,
            device="cuda",
            use_flash_attention=True,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            quantization=None,
        )
        log.info("DiT init: %s", init_status)

        log.info("Loading LM model: %s", LM_MODEL)
        self.llm_handler = LLMHandler()
        lm_status, lm_success = self.llm_handler.initialize(
            checkpoint_dir=str(checkpoint_dir),
            lm_model_path=LM_MODEL,
            backend="pt",
            device="cuda",
            offload_to_cpu=False,
        )
        log.info("LLM init: %s (success=%s)", lm_status, lm_success)

    def predict(
        self,
        prompt: str = Input(
            description="Short text describing the desired music — genre, mood, instruments, style. Max 512 characters.",
            default="upbeat electronic dance music with heavy bass and synth leads",
        ),
        lyrics: str = Input(
            description="Lyrics for the song. Use '[Instrumental]' for instrumental tracks. Max 4096 characters.",
            default="[Instrumental]",
        ),
        duration: float = Input(
            description="Target audio length in seconds. Set to -1 for auto.",
            default=30.0,
            ge=-1.0,
            le=600.0,
        ),
        bpm: Optional[int] = Input(
            description="Beats per minute (30-300). Leave unset for auto-detection by the LM.",
            default=None,
            ge=30,
            le=300,
        ),
        key_scale: str = Input(
            description="Musical key and scale (e.g. 'C major', 'F# minor', 'Bb major'). Leave empty for auto.",
            default="",
        ),
        time_signature: str = Input(
            description="Time signature: 2 for 2/4, 3 for 3/4, 4 for 4/4, 6 for 6/8. Use 'auto' for auto-detection.",
            default="auto",
            choices=["auto", "2", "3", "4", "6"],
        ),
        inference_steps: int = Input(
            description="Number of diffusion steps. Turbo model: 4-8 recommended. Base/SFT: 32-100.",
            default=8,
            ge=1,
            le=200,
        ),
        guidance_scale: float = Input(
            description="CFG strength. Only used by base/SFT models — ignored by turbo. Higher = follows prompt more strictly.",
            default=7.0,
            ge=1.0,
            le=15.0,
        ),
        shift: float = Input(
            description="Timestep shift factor. Default 1.0, use 3.0 for turbo model.",
            default=3.0,
            ge=1.0,
            le=5.0,
        ),
        seed: int = Input(
            description="Random seed for reproducibility. -1 for random.",
            default=-1,
        ),
        thinking: bool = Input(
            description="Enable LM chain-of-thought reasoning for metadata, caption, and language detection.",
            default=True,
        ),
        batch_size: int = Input(
            description="Number of songs to generate in parallel.",
            default=1,
            ge=1,
            le=4,
        ),
        audio_format: str = Input(
            description="Output audio format.",
            default="flac",
            choices=["mp3", "wav", "flac"],
        ),
    ) -> List[Path]:
        output_dir = tempfile.mkdtemp()

        params = GenerationParams(
            caption=prompt,
            lyrics=lyrics,
            duration=duration,
            bpm=bpm,
            keyscale=key_scale,
            timesignature="" if time_signature == "auto" else time_signature,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            seed=seed,
            thinking=thinking,
            task_type="text2music",
            infer_method="ode",
        )

        config = GenerationConfig(
            batch_size=batch_size,
            use_random_seed=(seed < 0),
            seeds=[seed] * batch_size if seed >= 0 else None,
            audio_format=audio_format,
        )

        result = generate_music(
            self.dit_handler,
            self.llm_handler,
            params,
            config,
            save_dir=output_dir,
        )

        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")

        output_paths = []
        for audio in result.audios:
            audio_path = audio.get("path")
            if audio_path and os.path.exists(audio_path):
                output_paths.append(Path(audio_path))

        if not output_paths:
            raise RuntimeError("No audio files were generated")

        return output_paths
