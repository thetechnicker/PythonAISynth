# cython: language_level=3
cimport numpy as np
import torch
cimport torch
from libc.stdlib cimport malloc, free



cpdef wrap_concat(tensor, idx1, idx2):
    length = tensor.size(0)

    if idx2 >= length:
        idx2 = idx2 % length

    if idx1 <= idx2:
        result = tensor[idx1:idx2]
    else:
        result = torch.cat((tensor[idx1:], tensor[:idx2]))

    return result


cdef class AudioCallback:
    cdef public object fourier_nn
    cdef public object t2_tensor
    cdef public int max_parallel_notes
    cdef public int samplerate
    cdef int i

    def __init__(self, fourier_nn, t2_tensor, max_parallel_notes, samplerate):
        self.fourier_nn=fourier_nn
        self.t2_tensor = t2_tensor
        self.max_parallel_notes = max_parallel_notes
        self.samplerate = samplerate
        self.i = 0

    cpdef void audio_callback(self, np.ndarray outdata, int frames, time, status):
        cdef int j
        a = torch.zeros((self.max_parallel_notes, frames, 1), device=self.fourier_nn.device)

        if status:
            print(status)

        with torch.no_grad():
            for j in range(self.max_parallel_notes):
                a[j, :] = self.fourier_nn.current_model(wrap_concat(self.t2_tensor[j], self.i, self.i + frames))

        y = torch.clamp(torch.sum(a, dim=0), min=-1, max=1).cpu().numpy()
        outdata[:] = y.reshape(-1, 1)
        self.i = (self.i + frames) % self.samplerate
