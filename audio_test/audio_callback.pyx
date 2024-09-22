# cython: language_level=3
cimport numpy as np
cimport torch
from libc.stdlib cimport malloc, free
from fourier_nn_template cimport FourierNNTemplate  # Import the template class


cdef wrap_concat(tensor, idx1, idx2):
    length = tensor.size(0)

    if idx2 >= length:
        idx2 = idx2 % length

    if idx1 <= idx2:
        result = torch.cat((tensor[idx1:], tensor[:idx2]))

    return result

cdef void audio_callback(np.ndarray outdata, int frames, time, status, FourierNNTemplate fourier_nn, torch.Tensor t2_tensor, int max_parralel_notes, int samplerate):
    cdef int i = 0
    cdef torch.Tensor a = torch.zeros((max_parralel_notes, frames, 1), device=fourier_nn.device)
    cdef torch.Tensor y

    if status:
        print(status)

    with torch.no_grad():
        for j in range(max_parralel_notes):
            a[j, :] = fourier_nn.current_model(wrap_concat(t2_tensor[j], i, i + frames))

    y = torch.clamp(torch.sum(a, dim=0), min=-1, max=1).cpu().numpy()
    outdata[:] = y.reshape(-1, 1)
    i = (i + frames) % samplerate
