import os
import sys
import site
import ctypes
import logging
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ModelResource:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.initialized_device = None
        self._session = None
        self._input_name = None

    def _setup_env(self):
        platform = sys.platform
        site_paths = list(site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            site_paths.append(user_site)

        if platform == "linux":
            self._init_linux_libs(site_paths)
        elif platform == "win32":
            self._init_windows_libs(site_paths)

    def _init_linux_libs(self, paths: list[str]):
        libs = [
            ("cudnn", "lib", "libcudnn.so.9"),
            ("cublas", "lib", "libcublas.so.12"),
        ]
        for p in paths:
            for pkg, sub, libname in libs:
                lib_path = os.path.join(p, "nvidia", pkg, sub, libname)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                        logger.debug(f"Loaded Linux lib: {libname}")
                    except Exception as e:
                        logger.warning(f"Failed to load {libname} from {p}: {e}")

    def _init_windows_libs(self, paths: list[str]):
        for p in paths:
            base_nv = os.path.join(p, "nvidia")
            if not os.path.exists(base_nv):
                continue
            for root, dirs, _ in os.walk(base_nv):
                target = "bin" if "bin" in dirs else ("lib" if "lib" in dirs else None)
                if target:
                    dll_dir = os.path.join(root, target)
                    os.add_dll_directory(dll_dir)
                    logger.debug(f"Added Windows DLL directory: {dll_dir}")

    def get_session(self) -> ort.InferenceSession:
        if self._session is not None:
            return self._session

        self._setup_env()

        available = ort.get_available_providers()
        providers = []

        if self.device in ("cuda", "auto"):
            if "TensorRTExecutionProvider" in available:
                providers.append("TensorRTExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")

        try:
            self._session = ort.InferenceSession(self.model_path, providers=providers)
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
