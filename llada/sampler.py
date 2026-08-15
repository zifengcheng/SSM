# Copyright 2025 Xinhua Chen, Duke CEI Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import List

class Sampler:
    def __init__(self, length=256, window=None,**kargs):
        self.length = length
        self.window = window
        self.kargs = kargs    
        self._pdf = None

        # assert self.window <= self.length
        
    def pdf():
        pass
       
    def sample(self, src: torch.Tensor):
        '''
        rejection sampling
        '''

        uniform = torch.rand(src.shape[0], device=src.device)  # Generate uniform random numbers on the same device as src
        # print("look: ", len(self.pdf()[:src.shape[0]]), self.pdf()[:src.shape[0]])
        return src[uniform < self.pdf()[:src.shape[0]]] 


class GaussianSampler(Sampler):
    def __init__(self, length=256, window=None,sigma=1.0, scale=1.0):
        super().__init__(length,window)
        self.sigma = sigma
        self.scale = scale

    
    def pdf(self):
        '''
        Generate Gaussian PDF values.
        '''
        if self._pdf is not None:
            return self._pdf
        mean = 0.0  
        std_dev = 1

        x = torch.linspace(mean, mean + self.sigma * std_dev, self.window, device='cuda')

        self._pdf = self.scale * torch.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * torch.sqrt(torch.tensor(2 * torch.pi)))
        extended_pdf = torch.zeros(self.length-self.window, device='cuda')
        self._pdf = torch.cat([self._pdf, extended_pdf])

        return self._pdf
    
class UniformSampler(Sampler):
    def __init__(self, length=256, window=None, number=0):
        super().__init__(length,window)
        self.number = number
        assert self.number <= self.length
    

    def sample(self, src: torch.Tensor):
        if self.number >= src.shape[0]:
            return src
        
        indices = torch.sort(torch.randperm(min(src.shape[0],self.window))[:self.number]).values
       
        return src[indices]

# Version 1
class SSMStructuredSampler(Sampler):
    """
    Structured SSM sampler

    Keep a strong contiguous local suffix region, then add a very small
    structured skeleton from far suffix:
    1) hard local window
    2) optional strided anchors in remaining suffix
    3) optional suffix-end anchor
    4) optional future block-boundary anchors
    """
    def __init__(
        self,
        length=256,
        window=None,
        local_window=128,
        num_anchors=16,
        block_size=32,
        use_suffix_end=True,
        use_block_boundaries=True,
        block_boundary_mode="start",
        block_boundary_offset=0,
        **kwargs,
    ):
        super().__init__(length, window)
        self.local_window = local_window
        self.num_anchors = num_anchors
        self.block_size = block_size
        self.use_suffix_end = use_suffix_end
        self.use_block_boundaries = use_block_boundaries
        self.block_boundary_mode = str(block_boundary_mode)
        self.block_boundary_offset = int(block_boundary_offset)

    def _get_boundary_anchors(self, src: torch.Tensor, core_limit: int) -> torch.Tensor:
        """
        Keep one anchor per future block in far region.
        Modes:
          - start: block start
          - end: block end
                    - both: keep block start and block end
          - fixed: fixed offset inside each block
          - random: random offset inside each block
        Plus:
          - always keep the end token of the last future block
        """
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len <= core_limit:
            return torch.tensor([], dtype=torch.long, device=device)

        # Only future blocks in far region are considered.
        first_block = max(0, core_limit // self.block_size)
        last_block = max(0, (suffix_len - 1) // self.block_size)
        if first_block > last_block:
            return torch.tensor([], dtype=torch.long, device=device)

        anchors = []
        for b in range(first_block, last_block + 1):
            b_start = b * self.block_size
            b_end = min((b + 1) * self.block_size - 1, suffix_len - 1)
            if b_end < core_limit:
                continue

            mode = self.block_boundary_mode
            if mode == "start":
                idx = b_start
                if idx >= core_limit:
                    anchors.append(idx)
            elif mode == "end":
                idx = b_end
                if idx >= core_limit:
                    anchors.append(idx)
            elif mode == "both":
                if b_start >= core_limit:
                    anchors.append(b_start)
                if b_end >= core_limit:
                    anchors.append(b_end)
            elif mode == "fixed":
                off = max(0, min(self.block_boundary_offset, self.block_size - 1))
                idx = min(b_start + off, b_end)
                if idx >= core_limit:
                    anchors.append(idx)
            elif mode == "random":
                span = max(1, b_end - b_start + 1)
                idx = b_start + int(torch.randint(low=0, high=span, size=(1,), device=device).item())
                if idx >= core_limit:
                    anchors.append(idx)
            else:
                raise ValueError(f"Unknown block_boundary_mode: {mode}")

        # Ensure the end token of the last future block is always kept.
        last_block_end = min((last_block + 1) * self.block_size - 1, suffix_len - 1)
        if last_block_end >= core_limit:
            anchors.append(last_block_end)

        if not anchors:
            return torch.tensor([], dtype=torch.long, device=device)
        anchors_t = torch.tensor(anchors, dtype=torch.long, device=device)
        anchors_t = torch.sort(torch.unique(anchors_t)).values
        return src[anchors_t]

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len == 0:
            return src

        # 1) hard local core
        core_limit = min(suffix_len, self.local_window)
        selected = [src[:core_limit]]

        # 2) far region candidates
        if core_limit < suffix_len:
            far_src = src[core_limit:]

            anchors = []

            # strided anchors (very small budget)
            if self.num_anchors > 0:
                steps = min(self.num_anchors, far_src.numel())
                strided_float = torch.linspace(0, far_src.numel() - 1, steps=steps, device=device)
                strided_local = torch.round(strided_float).long()
                strided_local = torch.unique(strided_local)
                anchors.append(far_src[strided_local])

            # future block boundaries
            if self.use_block_boundaries:
                boundary_points = self._get_boundary_anchors(src, core_limit)
                if boundary_points.numel() > 0:
                    anchors.append(boundary_points)

            # suffix-end anchor
            if self.use_suffix_end:
                anchors.append(src[-1:].to(device))

            if anchors:
                selected.append(torch.cat(anchors))

        final_indices = torch.cat(selected)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices


class SSMDropLastLocalTokenSampler(SSMStructuredSampler):
    """
    SSM control sampler:
    keep structured SSM behavior, except inside the local window:
    drop the last token of each local block, keep all other local tokens.

    Notes:
    - "local block" follows block_size partition on local indices [0, core_limit).
    - far-region policy (strided anchors / boundary anchors / suffix-end anchor)
      is unchanged from structured SSM.
    """

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len == 0:
            return src
        # Bugfix: when local window fully covers the suffix (strictly larger),
        # keep all suffix tokens and skip local block-tail dropout.
        if self.local_window > suffix_len:
            return src

        core_limit = min(suffix_len, self.local_window)

        # Local core with one-token-per-block tail drop:
        # drop indices i where (i + 1) % block_size == 0.
        local_idx = torch.arange(core_limit, device=device)
        if self.block_size > 1 and core_limit > 0:
            keep_mask = ((local_idx + 1) % self.block_size) != 0
            local_idx = local_idx[keep_mask]
        selected = [src[local_idx]]

        # Far region remains exactly the same as structured SSM.
        if core_limit < suffix_len:
            far_src = src[core_limit:]

            anchors = []

            if self.num_anchors > 0:
                steps = min(self.num_anchors, far_src.numel())
                strided_float = torch.linspace(0, far_src.numel() - 1, steps=steps, device=device)
                strided_local = torch.round(strided_float).long()
                strided_local = torch.unique(strided_local)
                anchors.append(far_src[strided_local])

            if self.use_block_boundaries:
                boundary_points = self._get_boundary_anchors(src, core_limit)
                if boundary_points.numel() > 0:
                    anchors.append(boundary_points)

            if self.use_suffix_end:
                anchors.append(src[-1:].to(device))

            if anchors:
                selected.append(torch.cat(anchors))

        final_indices = torch.cat(selected)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices


class SSMSampler(SSMStructuredSampler):
    """
    Base SSM block-structure sampler:
    - Local window policy:
      - mode='full': keep local window fully (main method default)
      - mode='none': keep no local token (local-window ablation)
    - For blocks after local window:
      - middle blocks (exclude last block):
        - mode='start_only': keep block-start only (main method)
        - mode='none': keep nothing (ablate middle region)
        - mode='end_only': keep block-end only
        - mode='start_end': keep block-start and block-end
        - mode='mid_only': keep block-middle only
      - last block:
        - mode='start_end': keep both start and end
        - mode='start_middle_end': keep start, middle and end
        - mode='end_only': keep end only
        - mode='start_only': keep start only
        - mode='none': keep nothing (full ablation on last block)
    """

    def __init__(
        self,
        length=256,
        window=None,
        local_window=128,
        num_anchors=16,
        block_size=32,
        use_suffix_end=True,
        use_block_boundaries=True,
        block_boundary_mode="start",
        block_boundary_offset=0,
        local_window_mode="full",
        local_middle_block_mode="start_only",
        local_last_block_mode="start_end",
        **kwargs,
    ):
        super().__init__(
            length=length,
            window=window,
            local_window=local_window,
            num_anchors=num_anchors,
            block_size=block_size,
            use_suffix_end=use_suffix_end,
            use_block_boundaries=use_block_boundaries,
            block_boundary_mode=block_boundary_mode,
            block_boundary_offset=block_boundary_offset,
            **kwargs,
        )
        local_mode = str(local_window_mode)
        if local_mode not in {"full", "none"}:
            raise ValueError(f"Unknown local_window_mode: {local_mode}")
        self.local_window_mode = local_mode

        middle_mode = str(local_middle_block_mode)
        if middle_mode not in {"start_only", "none", "end_only", "start_end", "mid_only"}:
            raise ValueError(f"Unknown local_middle_block_mode: {middle_mode}")
        self.local_middle_block_mode = middle_mode

        last_mode = str(local_last_block_mode)
        if last_mode not in {"start_end", "start_middle_end", "end_only", "start_only", "none"}:
            raise ValueError(f"Unknown local_last_block_mode: {last_mode}")
        self.local_last_block_mode = last_mode

    def _build_local_indices(self, core_limit: int, device: torch.device) -> torch.Tensor:
        if core_limit <= 0:
            return torch.tensor([], dtype=torch.long, device=device)

        if self.local_window_mode == "full":
            return torch.arange(core_limit, device=device)

        return torch.tensor([], dtype=torch.long, device=device)

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len == 0:
            return src

        core_limit = min(suffix_len, self.local_window)
        selected = []
        local_idx = self._build_local_indices(core_limit, device)
        if local_idx.numel() > 0:
            selected.append(src[local_idx])

        # After local window:
        # - middle blocks follow local_middle_block_mode
        # - last block follows local_last_block_mode
        if core_limit < suffix_len:
            bs = max(1, int(self.block_size))
            first_block = core_limit // bs
            last_block = (suffix_len - 1) // bs
            keep_far = []

            for b in range(first_block, last_block + 1):
                b_start = b * bs
                b_end = min((b + 1) * bs - 1, suffix_len - 1)
                if b_end < core_limit:
                    continue
                is_last = (b == last_block)
                if not is_last:
                    if self.local_middle_block_mode == "start_only":
                        keep_far.append(b_start)
                    elif self.local_middle_block_mode == "none":
                        pass
                    elif self.local_middle_block_mode == "end_only":
                        keep_far.append(b_end)
                    elif self.local_middle_block_mode == "start_end":
                        keep_far.append(b_start)
                        keep_far.append(b_end)
                    elif self.local_middle_block_mode == "mid_only":
                        b_mid = (b_start + b_end) // 2
                        keep_far.append(b_mid)
                else:
                    if self.local_last_block_mode == "start_end":
                        keep_far.append(b_start)
                        keep_far.append(b_end)
                    elif self.local_last_block_mode == "start_middle_end":
                        b_mid = (b_start + b_end) // 2
                        keep_far.append(b_start)
                        keep_far.append(b_mid)
                        keep_far.append(b_end)
                    elif self.local_last_block_mode == "end_only":
                        keep_far.append(b_end)
                    elif self.local_last_block_mode == "start_only":
                        keep_far.append(b_start)
                    elif self.local_last_block_mode == "none":
                        pass

            if keep_far:
                far_idx = torch.tensor(keep_far, dtype=torch.long, device=device)
                far_idx = torch.sort(torch.unique(far_idx)).values
                selected.append(src[far_idx])

        final_indices = torch.cat(selected) if selected else torch.tensor([], dtype=torch.long, device=device)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices


class SSMLocalStrideMidStartLastStartEndSampler(SSMStructuredSampler):
    """
    SSM sampler:
    - Local window: keep one token every `local_stride` positions.
    - Blocks after local window:
      - middle blocks: keep block-start only
      - last block: keep block-start and block-end
    """

    def __init__(
        self,
        length=256,
        window=None,
        local_window=128,
        local_stride=2,
        block_size=32,
        **kwargs,
    ):
        super().__init__(
            length=length,
            window=window,
            local_window=local_window,
            block_size=block_size,
            **kwargs,
        )
        self.local_stride = max(1, int(local_stride))

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len == 0:
            return src

        core_limit = min(suffix_len, self.local_window)
        selected = []

        if core_limit > 0:
            local_idx = torch.arange(0, core_limit, self.local_stride, device=device)
            selected.append(src[local_idx])

        if core_limit < suffix_len:
            bs = max(1, int(self.block_size))
            first_block = core_limit // bs
            last_block = (suffix_len - 1) // bs
            keep_far = []

            for b in range(first_block, last_block + 1):
                b_start = b * bs
                b_end = min((b + 1) * bs - 1, suffix_len - 1)
                if b_end < core_limit:
                    continue
                if b == last_block:
                    keep_far.append(b_start)
                    keep_far.append(b_end)
                else:
                    keep_far.append(b_start)

            if keep_far:
                far_idx = torch.tensor(keep_far, dtype=torch.long, device=device)
                far_idx = torch.sort(torch.unique(far_idx)).values
                selected.append(src[far_idx])

        final_indices = torch.cat(selected) if selected else torch.tensor([], dtype=torch.long, device=device)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices


class SSMLocalFullMidStartPlusOneLastStartEndSampler(SSMStructuredSampler):
    """
    SSM sampler:
    - Local window: keep all tokens (same as the structured SSM local core).
    - Blocks after local window:
      - middle blocks: keep block-start and block-start+1
      - last block: keep block-start and block-end
    """

    def __init__(
        self,
        length=256,
        window=None,
        local_window=128,
        block_size=32,
        **kwargs,
    ):
        super().__init__(
            length=length,
            window=window,
            local_window=local_window,
            block_size=block_size,
            **kwargs,
        )

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len == 0:
            return src

        core_limit = min(suffix_len, self.local_window)
        selected = []

        if core_limit > 0:
            selected.append(src[:core_limit])

        if core_limit < suffix_len:
            bs = max(1, int(self.block_size))
            first_block = core_limit // bs
            last_block = (suffix_len - 1) // bs
            keep_far = []

            for b in range(first_block, last_block + 1):
                b_start = b * bs
                b_end = min((b + 1) * bs - 1, suffix_len - 1)
                if b_end < core_limit:
                    continue
                if b == last_block:
                    keep_far.append(b_start)
                    keep_far.append(b_end)
                else:
                    keep_far.append(b_start)
                    keep_far.append(min(b_start + 1, b_end))

            if keep_far:
                far_idx = torch.tensor(keep_far, dtype=torch.long, device=device)
                far_idx = torch.sort(torch.unique(far_idx)).values
                selected.append(src[far_idx])

        final_indices = torch.cat(selected) if selected else torch.tensor([], dtype=torch.long, device=device)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices


class SSMLocalPrefixMidStartLastStartEndSampler(SSMStructuredSampler):
    """
    SSM sampler:
    - Local window: for each local block keep the first floor(block_size * b) tokens.
    - Blocks after local window:
      - middle blocks: keep block-start only
      - last block: keep block-start and block-end
    """

    def __init__(
        self,
        length=256,
        window=None,
        local_window=128,
        local_block_keep_ratio=0.8,
        block_size=32,
        **kwargs,
    ):
        super().__init__(
            length=length,
            window=window,
            local_window=local_window,
            block_size=block_size,
            **kwargs,
        )
        self.local_block_keep_ratio = float(local_block_keep_ratio)
        if self.local_block_keep_ratio < 0.0 or self.local_block_keep_ratio > 1.0:
            raise ValueError(f"local_block_keep_ratio must be in [0, 1], got {local_block_keep_ratio}")

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len == 0:
            return src

        core_limit = min(suffix_len, self.local_window)
        selected = []

        if core_limit > 0:
            bs = max(1, int(self.block_size))
            keep_per_block = int(bs * self.local_block_keep_ratio)
            if self.local_block_keep_ratio > 0.0 and keep_per_block == 0:
                keep_per_block = 1

            keep_local = []
            first_local_block = 0
            last_local_block = (core_limit - 1) // bs
            for b in range(first_local_block, last_local_block + 1):
                b_start = b * bs
                b_end_exclusive = min((b + 1) * bs, core_limit)
                block_len = b_end_exclusive - b_start
                if block_len <= 0:
                    continue
                k = min(block_len, keep_per_block)
                if k <= 0:
                    continue
                keep_local.append(torch.arange(b_start, b_start + k, device=device))

            if keep_local:
                local_idx = torch.cat(keep_local)
                selected.append(src[local_idx])

        if core_limit < suffix_len:
            bs = max(1, int(self.block_size))
            first_block = core_limit // bs
            last_block = (suffix_len - 1) // bs
            keep_far = []

            for b in range(first_block, last_block + 1):
                b_start = b * bs
                b_end = min((b + 1) * bs - 1, suffix_len - 1)
                if b_end < core_limit:
                    continue
                if b == last_block:
                    keep_far.append(b_start)
                    keep_far.append(b_end)
                else:
                    keep_far.append(b_start)

            if keep_far:
                far_idx = torch.tensor(keep_far, dtype=torch.long, device=device)
                far_idx = torch.sort(torch.unique(far_idx)).values
                selected.append(src[far_idx])

        final_indices = torch.cat(selected) if selected else torch.tensor([], dtype=torch.long, device=device)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices


class SSMLocalAltMiddleStartLastStartEndSampler(SSMStructuredSampler):
    """
    SSM block-structure control sampler:
    - Keep local window fully (same as the structured SSM local core).
    - For blocks after local window:
      - middle blocks (exclude last block): keep one block-start every other block
      - last block: keep start and end
    """

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len == 0:
            return src

        core_limit = min(suffix_len, self.local_window)
        selected = [src[:core_limit]]

        if core_limit < suffix_len:
            bs = max(1, int(self.block_size))
            first_block = core_limit // bs
            last_block = (suffix_len - 1) // bs
            keep_far = []

            for b in range(first_block, last_block + 1):
                b_start = b * bs
                b_end = min((b + 1) * bs - 1, suffix_len - 1)
                if b_end < core_limit:
                    continue
                is_last = (b == last_block)
                if is_last:
                    keep_far.append(b_start)
                    keep_far.append(b_end)
                else:
                    # Keep one start token every other middle block.
                    if ((b - first_block) % 2) == 0:
                        keep_far.append(b_start)

            if keep_far:
                far_idx = torch.tensor(keep_far, dtype=torch.long, device=device)
                far_idx = torch.sort(torch.unique(far_idx)).values
                selected.append(src[far_idx])

        final_indices = torch.cat(selected)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices


class SSMNoPenultStartSampler(SSMStructuredSampler):
    """
    SSM control sampler:
    keep structured SSM behavior, except one rule in boundary anchors:
    for the penultimate future block, do NOT keep its block-start anchor.

    Effect by boundary mode:
      - start: penultimate block contributes nothing
      - both: penultimate block keeps end only
      - end/fixed/random: unchanged
    """

    def _get_boundary_anchors(self, src: torch.Tensor, core_limit: int) -> torch.Tensor:
        suffix_len = src.shape[0]
        device = src.device
        if suffix_len <= core_limit:
            return torch.tensor([], dtype=torch.long, device=device)

        first_block = max(0, core_limit // self.block_size)
        last_block = max(0, (suffix_len - 1) // self.block_size)
        if first_block > last_block:
            return torch.tensor([], dtype=torch.long, device=device)

        penult_block = last_block - 1
        anchors = []
        for b in range(first_block, last_block + 1):
            b_start = b * self.block_size
            b_end = min((b + 1) * self.block_size - 1, suffix_len - 1)
            if b_end < core_limit:
                continue

            is_penult = (b == penult_block)
            mode = self.block_boundary_mode
            if mode == "start":
                # control rule: skip start anchor for penultimate future block
                if (not is_penult) and b_start >= core_limit:
                    anchors.append(b_start)
            elif mode == "end":
                if b_end >= core_limit:
                    anchors.append(b_end)
            elif mode == "both":
                # control rule: drop start only for penultimate block, keep end normally
                if (not is_penult) and b_start >= core_limit:
                    anchors.append(b_start)
                if b_end >= core_limit:
                    anchors.append(b_end)
            elif mode == "fixed":
                off = max(0, min(self.block_boundary_offset, self.block_size - 1))
                idx = min(b_start + off, b_end)
                if idx >= core_limit:
                    anchors.append(idx)
            elif mode == "random":
                span = max(1, b_end - b_start + 1)
                idx = b_start + int(torch.randint(low=0, high=span, size=(1,), device=device).item())
                if idx >= core_limit:
                    anchors.append(idx)
            else:
                raise ValueError(f"Unknown block_boundary_mode: {mode}")

        if not anchors:
            return torch.tensor([], dtype=torch.long, device=device)
        anchors_t = torch.tensor(anchors, dtype=torch.long, device=device)
        anchors_t = torch.sort(torch.unique(anchors_t)).values
        return src[anchors_t]


class SSMLastBlockFullNoAnchorsSampler(Sampler):
    """
    SSM control sampler:
    1) keep SSM local window (contiguous near suffix region)
    2) keep the last full block in suffix
    3) no strided/suffix-end-only anchors
    4) optional boundary anchors with the same config interface as SSM
    """

    def __init__(
        self,
        length=256,
        window=None,
        local_window=128,
        block_size=32,
        use_block_boundaries=False,
        block_boundary_mode="start",
        block_boundary_offset=0,
        **kwargs,
    ):
        super().__init__(length, window)
        self.local_window = int(local_window)
        self.block_size = int(block_size)
        self.use_block_boundaries = bool(use_block_boundaries)
        self.block_boundary_mode = str(block_boundary_mode)
        self.block_boundary_offset = int(block_boundary_offset)

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        if suffix_len == 0:
            return src

        core_limit = min(suffix_len, self.local_window)
        selected = [src[:core_limit]]

        # Keep last full block in suffix as an explicit global tail context.
        tail_keep = min(self.block_size, suffix_len)
        selected.append(src[-tail_keep:])

        # Optional SSM-style future block-boundary anchors.
        if self.use_block_boundaries:
            boundary_points = SSMStructuredSampler._get_boundary_anchors(self, src, core_limit)
            if boundary_points.numel() > 0:
                selected.append(boundary_points)

        final_indices = torch.cat(selected)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices


class StreamingDLLMSampler(Sampler):
    """
    Streaming-DLLM style sampler (SSM-based control):
    1) Keep SSM local window (contiguous near suffix region)
    2) Keep suffix-end anchor (last suffix token)

    No far strided anchors and no boundary anchors.
    """

    def __init__(
        self,
        length=256,
        window=None,
        local_window=128,
        use_suffix_end=True,
        **kwargs,
    ):
        # This sampler does not use window; pin to length to satisfy base checks.
        super().__init__(length, length)
        self.local_window = int(local_window)
        self.use_suffix_end = bool(use_suffix_end)

    def sample(self, src: torch.Tensor):
        suffix_len = src.shape[0]
        if suffix_len == 0:
            return src

        core_limit = min(suffix_len, self.local_window)
        selected = [src[:core_limit]]

        # suffix-end anchor
        if self.use_suffix_end:
            selected.append(src[-1:])

        final_indices = torch.cat(selected)
        final_indices = torch.sort(torch.unique(final_indices)).values
        return final_indices
