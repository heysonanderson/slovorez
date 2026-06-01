import logging
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_AUTO_GPU_PROVIDERS = [
    "CUDAExecutionProvider",
]

ort.preload_dlls()

class ModelResource:
    """Wraps an ONNX Runtime inference session with lazy initialisation.

    Args:
        model_path: path to the ``.onnx`` weights file.
        device:     execution provider hint.

                    * ``"auto"``     -- try CUDA, fall back to CPU (default).
                    * ``"cuda"``     -- require CUDA; raise if unavailable.
                    * ``"tensorrt"`` -- try TensorRT → CUDA → CPU. Requires
                                       libnvinfer; raises if TensorRT provider
                                       is not registered in ORT.
                    * ``"cpu"``      -- CPU only.
    """

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.initialized_device: str | None = None
        self._session: ort.InferenceSession | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def _build_providers(self) -> list[str]:
        available = ort.get_available_providers()

        if self.device == "cpu":
            return ["CPUExecutionProvider"]

        if self.device == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError(
                    "Requested device='tensorrt', but TensorrtExecutionProvider is not "
                    "registered in this ORT build. Install TensorRT libraries and ensure "
                    "they are on LD_LIBRARY_PATH."
                )
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        cuda_providers = [p for p in _AUTO_GPU_PROVIDERS if p in available]

        if self.device == "cuda" and not cuda_providers:
            raise RuntimeError(
                f"Requested device='cuda', but CUDAExecutionProvider is not available. "
                f"Registered providers: {available}. "
                "Check CUDA / cuDNN installation and LD_LIBRARY_PATH."
            )

        return list(dict.fromkeys(cuda_providers + ["CPUExecutionProvider"]))

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def get_session(self) -> ort.InferenceSession:
        """Return the inference session, creating it on first call."""
        if self._session is not None:
            return self._session

        providers = self._build_providers()
        logger.debug("Creating ONNX session with providers: %s", providers)

        try:
            self._session = ort.InferenceSession(
                self.model_path,
                providers=providers,
            )
            
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            
            self.initialized_device = self._session.get_providers()[0]
            logger.info("Session initialized with provider: %s", self.initialized_device)
        except Exception as e:
            logger.error("Failed to create ONNX session: %s", e)
            raise

        return self._session

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, *args: np.ndarray, **kwargs: np.ndarray) -> np.ndarray | list[np.ndarray]:
        session = self.get_session()
        
        feed_dict = kwargs.copy()

        for i, arg in enumerate(args):
            if i >= len(self._input_names):
                raise ValueError(
                    f"Passed {len(args)} positional arguments, but the model "
                    f"accepts a maximum of {len(self._input_names)} inputs."
                )
            name = self._input_names[i]
            if name in feed_dict:
                raise ValueError(f"Duplicate value assigned to input '{name}' via both args and kwargs.")
            feed_dict[name] = arg

        outputs = session.run(None, feed_dict)

        return outputs[0] if len(outputs) == 1 else outputs