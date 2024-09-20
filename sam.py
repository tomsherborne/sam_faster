import torch
from torch.optim.optimizer import Optimizer

def group_tensors_by_device_and_dtype(tensors_lists):
    grouped_tensors = {}
    for tensors in zip(*tensors_lists):
        devices = [t.device for t in tensors]
        dtypes = [t.dtype for t in tensors]
        if len(set(devices)) != 1 or len(set(dtypes)) != 1:
            raise ValueError("All tensors in the group must be on the same device and have the same dtype")
        key = (devices[0], dtypes[0])
        if key not in grouped_tensors:
            grouped_tensors[key] = [[] for _ in tensors_lists]
        for i, t in enumerate(tensors):
            grouped_tensors[key][i].append(t)
    return grouped_tensors

class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            # Imitate Adam._init_group() but simpler
            params_with_grad = []
            grads = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    self.state[p]["p_old"] = p.data.clone()

            # Use the custom grouping function
            grouped_tensors = group_tensors_by_device_and_dtype(
                [params_with_grad, grads]
            )

            # Process each group of tensors
            for (device_params, device_grads) in grouped_tensors.values():
                # Handle complex parameters
                device_grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_grads]
                device_params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in device_params]
                device_scale = scale.clone().to(device_params[0])
                if group["adaptive"]:
                    e_w = torch._foreach_mul(device_params, device_params)
                else:
                    e_w = [torch.ones_like(p.grad) for p in device_params]

                torch._foreach_mul_(e_w, device_scale)
                torch._foreach_add_(device_params, e_w)
                
                del e_w, device_grads, device_scale

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["p_old"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
