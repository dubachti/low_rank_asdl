from contextlib import contextmanager

import torch
from torch import eig, nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, Subset, DataLoader
from typing import List, Callable
#from torch.cuda import nvtx

_REQUIRES_GRAD_ATTR = '_original_requires_grad'

__all__ = [
    'original_requires_grad', 'record_original_requires_grad',
    'restore_original_requires_grad', 'skip_param_grad', 'im2col_2d',
    'im2col_2d_slow', 'cholesky_inv', 'sherman_morrison_inv',
    'power_method', 'PseudoBatchLoaderGenerator'
]


def original_requires_grad(module=None, param_name=None, param=None):
    if param is None:
        assert module is not None and param_name is not None
        param = getattr(module, param_name, None)
    return param is not None and getattr(param, _REQUIRES_GRAD_ATTR)


def record_original_requires_grad(param):
    setattr(param, _REQUIRES_GRAD_ATTR, param.requires_grad)


def restore_original_requires_grad(param):
    param.requires_grad = getattr(param, _REQUIRES_GRAD_ATTR,
                                  param.requires_grad)


@contextmanager
def skip_param_grad(model, disable=False):
    if not disable:
        for param in model.parameters():
            record_original_requires_grad(param)
            param.requires_grad = False

    yield
    if not disable:
        for param in model.parameters():
            restore_original_requires_grad(param)


def im2col_2d(x: torch.Tensor, conv2d: nn.Module):
    assert x.ndimension() == 4  # n x c x h_in x w_in
    assert isinstance(conv2d, (nn.Conv2d, nn.ConvTranspose2d))
    assert conv2d.dilation == (1, 1)

    ph, pw = conv2d.padding
    kh, kw = conv2d.kernel_size
    sy, sx = conv2d.stride
    if ph + pw > 0:
        x = F.pad(x, (pw, pw, ph, ph)).data
    x = x.unfold(2, kh, sy)  # n x c x h_out x w_in x kh
    x = x.unfold(3, kw, sx)  # n x c x h_out x w_out x kh x kw
    x = x.permute(0, 1, 4, 5, 2,
                  3).contiguous()  # n x c x kh x kw x h_out x w_out
    x = x.view(x.size(0),
               x.size(1) * x.size(2) * x.size(3),
               x.size(4) * x.size(5))  # n x c(kh)(kw) x (h_out)(w_out)
    return x


def im2col_2d_slow(x: torch.Tensor, conv2d: nn.Module):
    assert x.ndimension() == 4  # n x c x h_in x w_in
    assert isinstance(conv2d, (nn.Conv2d, nn.ConvTranspose2d))

    # n x c(k_h)(k_w) x (h_out)(w_out)
    Mx = F.unfold(x,
                  conv2d.kernel_size,
                  dilation=conv2d.dilation,
                  padding=conv2d.padding,
                  stride=conv2d.stride)

    return Mx


def cholesky_inv(X, damping=1e-7):
    diag = torch.diagonal(X)
    diag += damping
    u = torch.linalg.cholesky(X)
    diag -= damping
    return torch.cholesky_inverse(u)

#@nvtx.range('sherman_morrison_inv')
def sherman_morrison_inv(eig: torch.Tensor,
                         vec: torch.Tensor,
                         damping: torch.Tensor,
                         diag: torch.Tensor = None,
                         device: torch.device = None):               

    dim = vec.shape[1]
    if diag is None: diag = torch.tensor(0, device=device)
    D_inv = 1/(torch.ones(dim, device=device) * damping + diag)
    top = eig[0] * torch.outer(torch.mul(D_inv, vec[0]), torch.mul(vec[0], D_inv))
    bot = 1. + (eig[0] * torch.dot(torch.mul(vec[0], D_inv), vec[0]))
    inv = torch.diag(D_inv) - (top/bot)
    for w, v in zip(eig[1:], vec[1:]):
        top = w * torch.outer(torch.matmul(inv, v), torch.matmul(torch.t(v), inv))
        bot = 1. + (w * torch.dot(torch.matmul(torch.t(v), inv), v))
        inv = inv - (top/bot)
    return inv

#@nvtx.range('power_method')
def power_method(mvp_fn,
                shape,
                top_n,
                max_itr,
                device,
                tol=1e-6,
                random_seed=None):

    assert top_n >= 1, f'rank {top_n} not possible'
    assert max_itr >= 1, f'max_iters = {max_itr} not possible'

    if top_n > min(shape): top_n = min(shape)

    device = torch.device('cpu')

    eigvals = []
    eigvecs = []

    for i in range(top_n):

        vec = torch.rand(shape[1], device='cpu')
        #vec = torch.ones(shape[1], device='cpu')

        eigval = None
        last_eigval = None
        # power iteration
        for j in range(max_itr):
            vec = _orthonormal(vec, eigvecs)
            Mv = _mvp(mvp_fn, vec, random_seed=random_seed)
            eigval = Mv.dot(vec)
            if j > 0:
                diff = torch.abs(eigval - last_eigval) / (torch.abs(last_eigval) + 1e-5)
                if diff < tol:
                    break
            last_eigval = eigval
            vec = Mv

        ####
        #with open(f'iterations_{max_itr}.txt', 'a+') as f:
        #    f.write(f'{j+1} \n')
        ####

        eigvals.append(eigval)
        eigvecs.append(vec)
    
    return torch.tensor(eigvals, device=device), torch.stack(eigvecs).to(device)

#@nvtx.range('_mvp')
def _mvp(mvp_fn: Callable[[torch.Tensor], torch.Tensor],
        vec: torch.Tensor,
        random_seed=None,
        damping=None) -> torch.Tensor:
    if random_seed:
        # for matrices that are not deterministic (e.g., fisher_mc)
        torch.manual_seed(random_seed)
    Mv = mvp_fn(vec)
    if damping:
        Mv.add_(vec, alpha=damping)
    return Mv

#@nvtx.range('_orthonotmal')
def _orthonormal(w: torch.Tensor, v_list: List[torch.Tensor]) -> torch.Tensor:
    for v in v_list:
        w = w.add(v, alpha=-w.dot(v))
    return torch.nn.functional.normalize(w, dim=0)

class PseudoBatchLoaderGenerator:
    """
    Example::
    >>> # create a base dataloader
    >>> dataset_size = 10
    >>> x_all = torch.tensor(range(dataset_size))
    >>> dataset = torch.utils.data.TensorDataset(x_all)
    >>> data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)
    >>>
    >>> # create a pseudo-batch loader generator
    >>> pb_loader_generator = PseudoBatchLoaderGenerator(data_loader, 5)
    >>>
    >>> for i, pb_loader in enumerate(pb_loader_generator):
    >>>     print(f'pseudo-batch at step {i}')
    >>>     print(list(pb_loader))

    Outputs:
    ```
    pseudo-batch at step 0
    [[tensor([0])], [tensor([1])], [tensor([3])], [tensor([6])], [tensor([7])]]
    pseudo-batch at step 1
    [[tensor([8])], [tensor([5])], [tensor([4])], [tensor([2])], [tensor([9])]]
    ```
    """
    def __init__(self,
                 base_data_loader,
                 pseudo_batch_size,
                 batch_size=None,
                 drop_last=None):
        if batch_size is None:
            batch_size = base_data_loader.batch_size
        assert pseudo_batch_size % batch_size == 0, f'pseudo_batch_size ({pseudo_batch_size}) ' \
                                                    f'needs to be divisible by batch_size ({batch_size})'
        if drop_last is None:
            drop_last = base_data_loader.drop_last
        base_dataset = base_data_loader.dataset
        sampler_cls = base_data_loader.sampler.__class__
        pseudo_batch_sampler = BatchSampler(sampler_cls(
            range(len(base_dataset))),
                                            batch_size=pseudo_batch_size,
                                            drop_last=drop_last)
        self.batch_size = batch_size
        self.pseudo_batch_sampler = pseudo_batch_sampler
        self.base_dataset = base_dataset
        self.base_data_loader = base_data_loader

    def __iter__(self):
        loader = self.base_data_loader
        for indices in self.pseudo_batch_sampler:
            subset_in_pseudo_batch = Subset(self.base_dataset, indices)
            data_loader = DataLoader(
                subset_in_pseudo_batch,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=loader.num_workers,
                collate_fn=loader.collate_fn,
                pin_memory=loader.pin_memory,
                drop_last=False,
                timeout=loader.timeout,
                worker_init_fn=loader.worker_init_fn,
                multiprocessing_context=loader.multiprocessing_context,
                generator=loader.generator,
                prefetch_factor=loader.prefetch_factor,
                persistent_workers=loader.persistent_workers)
            yield data_loader

    def __len__(self) -> int:
        return len(self.pseudo_batch_sampler)
