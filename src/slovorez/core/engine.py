import logging
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_PROVIDER_PRIORITY = [
    "TensorRTExecutionProvider",
    "CUDAExecutionProvider", 
    "CPUExecutionProvider",
]


class ModelResource:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.initialized_device: str | None = None
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

    def _build_providers(self) -> list[str]:
        available = ort.get_available_providers()

        if self.device == "cpu":
            return ["CPUExecutionProvider"]

        gpu_providers = [p for p in _PROVIDER_PRIORITY if p in available and p != "CPUExecutionProvider"]

        if self.device == "cuda":
            if not gpu_providers:
                raise RuntimeError(
                    f"Requested device='cuda', but no GPU providers found in available: {available}. "
                    "Check CUDA 12 / cuDNN 9 installation and PATH variables."
                )
            return list(dict.fromkeys(gpu_providers + ["CPUExecutionProvider"]))
    
        return list(dict.fromkeys(gpu_providers + available + ["CPUExecutionProvider"]))

    def get_session(self) -> ort.InferenceSession:
        if self._session is not None:
            return self._session

        providers = self._build_providers()
        try:
            self._session = ort.InferenceSession(
                self.model_path,
                providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self.initialized_device = self._session.get_providers()[0]
            logger.info(f"Session initialized with provider: {self.initialized_device}")
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise

        return self._session

    def predict(self, x: np.ndarray) -> np.ndarray:
        session = self.get_session()
        return session.run(None, {self._input_name: x})[0]