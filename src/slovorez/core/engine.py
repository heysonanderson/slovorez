from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_CPU_PROVIDER = "CPUExecutionProvider"
_VALID_DEVICES = frozenset({"auto", "cpu", "cuda", "tensorrt"})

_DEFAULT_CUDA_OPTIONS: dict[str, str | int | bool] = {
    "device_id": 0,
    "arena_extend_strategy": "kNextPowerOfTwo",
    "cudnn_conv_algo_search": "HEURISTIC",
    "do_copy_in_default_stream": True,
}


class ModelResource:
    """ONNX Runtime inference session with hardware-aware provider selection.

    Lazily initializes the session on first use — safe to instantiate before
    spawning worker processes.

    Args:
        model_path:   path to the ``.onnx`` weights file.
        device:       ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"tensorrt"``.
                      ``"auto"`` picks CUDA if available, otherwise CPU.
                      TRT is never selected automatically — pass ``"tensorrt"``
                      to opt in explicitly (requires TRT libraries installed).
        cuda_options: overrides for CUDA EP options (e.g. ``gpu_mem_limit``).
                      Merged on top of ``_DEFAULT_CUDA_OPTIONS``.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        *,
        cuda_options: dict | None = None,
    ):
        if device not in _VALID_DEVICES:
            raise ValueError(
                f"Unknown device={device!r}. "
                f"Expected one of: {', '.join(sorted(_VALID_DEVICES))}"
            )

        self.model_path = model_path
        self.device = device
        self.initialized_device: str | None = None

        self._cuda_options = {**_DEFAULT_CUDA_OPTIONS, **(cuda_options or {})}
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

    def _build_providers(self) -> list[tuple[str, dict] | str]:
        """Return an ordered provider list for ``InferenceSession``.

        TensorRT is excluded from ``"auto"`` because ORT reports it in
        ``get_available_providers()`` even when the TRT runtime libraries are
        not installed, which causes the entire provider chain to fail instead
        of falling back to CUDA.
        """
        available = set(ort.get_available_providers())

        if self.device == "cpu":
            return [_CPU_PROVIDER]

        if self.device == "tensorrt":
            providers: list[tuple[str, dict] | str] = []
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", self._cuda_options))
            providers.append(_CPU_PROVIDER)
            return providers

        # "cuda" or "auto"
        if "CUDAExecutionProvider" in available:
            return [("CUDAExecutionProvider", self._cuda_options), _CPU_PROVIDER]

        if self.device == "cuda":
            raise RuntimeError(
                "device='cuda' requested but CUDAExecutionProvider is not available. "
                f"Installed providers: {sorted(available)}. "
                "Check your CUDA / cuDNN installation and ensure "
                "onnxruntime-gpu is installed."
            )

        logger.info("No GPU provider available — using CPU.")
        return [_CPU_PROVIDER]

    def get_session(self) -> ort.InferenceSession:
        """Return the cached session, creating it on first call."""
        if self._session is not None:
            return self._session

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = self._build_providers()

        try:
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=providers,
            )
        except Exception as exc:
            if self.device in ("auto", "cuda"):
                logger.warning("GPU session failed (%s), retrying on CPU.", exc)
                self._session = ort.InferenceSession(
                    self.model_path, sess_options=opts, providers=[_CPU_PROVIDER],
                )
            else:
                raise

        self._input_name = self._session.get_inputs()[0].name
        active = self._session.get_providers()
        self.initialized_device = active[0] if active else _CPU_PROVIDER
        logger.info("ONNX session ready — provider: %s", self.initialized_device)
        return self._session

    @property
    def is_gpu(self) -> bool:
        """True if the session is running on a GPU provider."""
        return self.initialized_device not in (None, _CPU_PROVIDER)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run a forward pass. Input: int32 (batch, seq_len). Output: float32 logits."""
        session = self.get_session()
        return session.run(None, {self._input_name: x})[0]