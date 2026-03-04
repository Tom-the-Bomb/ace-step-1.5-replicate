import os
import tempfile
from pathlib import Path as PathLib
from typing import List

import logging

from cog import BasePredictor, Input, Path

log = logging.getLogger(__name__)

DIT_MODEL = "acestep-v15-turbo"
LM_MODEL = "acestep-5Hz-lm-4B"

class Predictor(BasePredictor):
    def setup(self) -> None:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.model_downloader import ensure_lm_model

        project_root = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = PathLib(project_root) / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        dl_ok, dl_msg = ensure_lm_model(
            model_name=LM_MODEL,
            checkpoints_dir=checkpoint_dir,
        )
        log.info("LM download: %s (ok=%s)", dl_msg, dl_ok)

        self.dit_handler = AceStepHandler()
        init_status, _ = self.dit_handler.initialize_service(
            project_root=project_root,
            config_path=DIT_MODEL,
            device="cuda",
            use_flash_attention=True,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
            quantization=None,
        )
        log.info("DiT init: %s", init_status)

        self.llm_handler = LLMHandler()
        lm_status, lm_success = self.llm_handler.initialize(
            checkpoint_dir=str(checkpoint_dir),
            lm_model_path=LM_MODEL,
            backend="vllm",
            device="cuda",
            offload_to_cpu=False,
        )
        log.info("LLM init: %s (success=%s)", lm_status, lm_success)

    def predict(
        self,
        prompt: str = Input(
            description="Text description of the music to generate. Include genre, mood, instruments, style.",
            default="upbeat electronic dance music with heavy bass and synth leads",
        ),
        lyrics: str = Input(
            description="Lyrics for the song. Use '[Instrumental]' for instrumental tracks. Supports 50+ languages.",
            default="[Instrumental]",
        ),
        duration: float = Input(
            description="Duration of the generated audio in seconds.",
            default=30.0,
            ge=10.0,
            le=240.0,
        ),
        bpm: int = Input(
            description="Beats per minute. Set to 0 for auto-detection by the LM.",
            default=0,
            ge=0,
            le=300,
        ),
        key_scale: str = Input(
            description="Musical key and scale (e.g. 'C major', 'A minor'). Leave empty for auto.",
            default="",
        ),
        time_signature: str = Input(
            description="Time signature.",
            default="4/4",
            choices=["2/4", "3/4", "4/4", "6/8"],
        ),
        inference_steps: int = Input(
            description="Number of denoising steps. Turbo model works best with 4-8 steps.",
            default=8,
            ge=1,
            le=20,
        ),
        guidance_scale: float = Input(
            description="Classifier-free guidance scale. Higher values follow the prompt more closely. Turbo model ignores this.",
            default=1.0,
            ge=1.0,
            le=15.0,
        ),
        shift: float = Input(
            description="Timestep shift factor. Recommended 3.0 for turbo model.",
            default=3.0,
            ge=1.0,
            le=5.0,
        ),
        seed: int = Input(
            description="Random seed for reproducibility. Use -1 for random.",
            default=-1,
        ),
        thinking: bool = Input(
            description="Enable LM chain-of-thought reasoning for better metadata generation.",
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
            default="mp3",
            choices=["mp3", "wav", "flac"],
        ),
    ) -> List[Path]:
        """Generate music from text description and optional lyrics."""
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        output_dir = tempfile.mkdtemp()

        params = GenerationParams(
            caption=prompt,
            lyrics=lyrics,
            duration=duration,
            bpm=bpm if bpm > 0 else None,
            keyscale=key_scale,
            timesignature=time_signature,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            seed=seed if seed >= 0 else -1,
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
