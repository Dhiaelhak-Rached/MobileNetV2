"""Shared TensorRT engine wrapper for the Jetson (used by validate_engine.py
and jetson_live_detect.py).

Uses PyTorch CUDA tensors as I/O buffers — both `torch` and `tensorrt` ship
with JetPack, so no extra CUDA bindings (pycuda / cuda-python) are needed.
Handles both TensorRT API generations: v10 (JetPack 6, tensor names) and
v8 (JetPack 5, binding indices).
"""
import numpy as np
import torch


class TrtEngine:
    """Minimal TensorRT runner: waveform [16000] float32 -> probabilities [2]."""

    def __init__(self, engine_path: str, n_samples: int = 16000):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self.x = torch.zeros(1, n_samples, dtype=torch.float32, device="cuda")
        self.y = torch.zeros(1, 2, dtype=torch.float32, device="cuda")
        self._v10 = hasattr(self.engine, "num_io_tensors")
        if self._v10:
            names = [self.engine.get_tensor_name(i)
                     for i in range(self.engine.num_io_tensors)]
            self.in_name = next(n for n in names
                                if self.engine.get_tensor_mode(n).name == "INPUT")
            self.out_name = next(n for n in names
                                 if self.engine.get_tensor_mode(n).name == "OUTPUT")

    def infer(self, waveform: np.ndarray) -> np.ndarray:
        """waveform: float32 [16000] -> probabilities [2] (index 1 = drone)."""
        self.x.copy_(torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)))
        if self._v10:
            self.ctx.set_tensor_address(self.in_name, self.x.data_ptr())
            self.ctx.set_tensor_address(self.out_name, self.y.data_ptr())
            self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        else:
            self.ctx.execute_v2([int(self.x.data_ptr()), int(self.y.data_ptr())])
        torch.cuda.synchronize()
        return self.y.cpu().numpy()[0]
