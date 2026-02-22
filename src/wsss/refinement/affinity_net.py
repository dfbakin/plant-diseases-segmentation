"""PSA affinity network: predicts pixel-pair semantic similarity.

Ported from MCTformer/psa/network/resnet38_aff.py. ResNet38 backbone with
three projection heads (conv4/5/6) merged into a 448-dim feature, then pairwise
affinity computed for pixel pairs within a local radius.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wsss.refinement.resnet38d import Net as ResNet38d


def get_indices_in_radius(height: int, width: int, radius: int) -> np.ndarray:
    """Compute (from, to) index pairs for all pixel pairs within radius."""
    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))
    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    full_indices = np.arange(0, height * width, dtype=np.int64).reshape(height, width)
    radius_floor = radius - 1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = full_indices[:-radius_floor, radius_floor:-radius_floor].reshape(-1)

    indices_from_to_list = []
    for dy, dx in search_dist:
        indices_to = full_indices[
            dy : dy + cropped_height, radius_floor + dx : radius_floor + dx + cropped_width
        ]
        indices_to = indices_to.reshape(-1)
        indices_from_to_list.append(np.stack((indices_from, indices_to), axis=1))

    return np.concatenate(indices_from_to_list, axis=0)


class AffinityNet(ResNet38d):
    """Affinity prediction on top of ResNet38d backbone."""

    def __init__(self, predefined_featuresize: int = 56):
        super().__init__()

        self.f8_3 = nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = nn.Conv2d(1024, 128, 1, bias=False)
        self.f8_5 = nn.Conv2d(4096, 256, 1, bias=False)
        self.f9 = nn.Conv2d(448, 448, 1, bias=False)

        nn.init.kaiming_normal_(self.f8_3.weight)
        nn.init.kaiming_normal_(self.f8_4.weight)
        nn.init.kaiming_normal_(self.f8_5.weight)
        nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_featuresize = predefined_featuresize
        ind = get_indices_in_radius(predefined_featuresize, predefined_featuresize, radius=5)
        self.register_buffer("ind_from", torch.from_numpy(ind[:, 0]), persistent=False)
        self.register_buffer("ind_to", torch.from_numpy(ind[:, 1]), persistent=False)

    def forward(self, x, to_dense=False):
        d = super().forward_as_dict(x)

        f8_3 = F.elu(self.f8_3(d["conv4"]))
        f8_4 = F.elu(self.f8_4(d["conv5"]))
        f8_5 = F.elu(self.f8_5(d["conv6"]))
        x = F.elu(self.f9(torch.cat([f8_3, f8_4, f8_5], dim=1)))

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind = get_indices_in_radius(x.size(2), x.size(3), 5)
            ind_from = torch.from_numpy(ind[:, 0]).to(x.device)
            ind_to = torch.from_numpy(ind[:, 1]).to(x.device)

        x = x.view(x.size(0), x.size(1), -1)

        ff = torch.index_select(x, dim=2, index=ind_from)
        ft = torch.index_select(x, dim=2, index=ind_to)

        ff = ff.unsqueeze(2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff = torch.exp(-torch.mean(torch.abs(ft - ff), dim=1))

        if to_dense:
            aff = aff.view(-1).cpu()
            ind_from_cpu = ind_from.cpu()
            ind_to_cpu = ind_to.cpu()
            ind_from_exp = ind_from_cpu.unsqueeze(0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to_cpu])
            indices_tp = torch.stack([ind_to_cpu, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = torch.sparse_coo_tensor(
                torch.cat([indices, indices_id, indices_tp], dim=1),
                torch.cat([aff, torch.ones([area]), aff]),
            ).to_dense()
            return aff_mat.to(x.device)

        return aff

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.modules.normalization.GroupNorm)):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)
                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups
