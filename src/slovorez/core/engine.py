from __future__ import annotations

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    try:
        import psutil
        affinity = psutil.Process().cpu_affinity()
        if affinity:
            return len(affinity)
    except Exception:
        pass
    return os.cpu_count() or 1


class ModelResource:
    """Wraps an ONNX Runtime inference session with lazy initialisation.

    Args:
        `model_path`: path to the ``.onnx`` weights file.
        `providers`: execution provider list, InferenceSession get it as is. `None` for `CPUExecutionProvider`.
        Other providers are not tested.
    """

    def __init__(self, model_path: str, providers: list[str] | None = None, intra = 1):
        self.model_path = model_path
        self.providers = ["CPUExecutionProvider"] if providers is None else providers
        self.intra: int = intra
        self._session: ort.InferenceSession | None = None       # type:ignore
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def get_session(self) -> ort.InferenceSession:                      # type:ignore
        """Return the inference session, creating it on first call."""
        if self._session is not None:
            return self._session
        import onnxruntime as ort
        providers = self.providers
        logger.debug("Creating ONNX session with providers: %s", providers)

        try:

            opts = ort.SessionOptions()
            if self.intra is not None:
                opts.intra_op_num_threads = self.intra
                opts.inter_op_num_threads = self.intra

            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=providers
            )
            
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            
            logger.info("Session initialized with provider: %s", self._session.get_providers()[0])
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