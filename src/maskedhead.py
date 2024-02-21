import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class MaskedRegressionHead(pl.LightningModule):
    def __init__(self, dim=512, fids=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(normalized_shape=[dim])
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fids = fids

    def mask_features(self, x, fids=None):
        ''' mask fids in calculation of feature attribution
            fids: feature_ids, list of ints'''
        if self.fids:
            fids = self.fids
        elif not fids:
            # fids = [int(torch.argmax(torch.abs(self.fc1.weight)))]
            # print([o for o in enumerate(torch.abs(self.fc1.weight).cpu().detach().numpy())])
            # [237, 196, 482, 145, 400, 323, 182, 379, 190, 445]
            vec = self.fc2.weight[0].cpu().detach().numpy()
            fids = [ix for ix, val in sorted(
                enumerate(np.abs(vec)),
                key=lambda a: a[1],
                reverse=True
            )]

            print(list(zip(fids[:10], vec[fids[:10]])))
            fids = fids[-2]
        self.fids = fids

        # print('feature ids to consider:', fids)
        mask = torch.zeros_like(x, dtype=torch.int64)
        mask[:, fids] = 1

        return x * mask

    def forward(self, x):
        ''' mask out all features excluding [fids]'''
        x = self.norm(x)

        # x = self.mask_features(x)
        x.register_hook(self.mask_features)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)


class MaskedLinearRegressionHead(pl.LightningModule):
    def __init__(self, dim=512, fids=None, sign=None):
        super().__init__()
        # self.norm = nn.LayerNorm(normalized_shape=[dim])
        self.dim = dim
        self.fc1 = nn.Linear(dim, 1, bias=False)
        self.fids = fids
        self.sign = sign
        print('masked head, sign: ', self.sign)

    def mask_quadrant(self, x):
        # activation mask
        a_sign = torch.sign(x)
        a_pos = torch.where(a_sign == 1, 1, 0)
        a_neg = torch.where(a_sign == -1, 1, 0)
        # weight vector mask
        w_sign = torch.sign(self.fc1.weight[0])
        w_pos = torch.where(w_sign == 1, 1, 0)
        w_neg = torch.where(w_sign == -1, 1, 0)
        print('act', torch.sum(a_sign, dim=-1), torch.sum(torch.abs(a_sign), dim=-1))
        print('wgt', torch.sum(w_sign), torch.sum(torch.abs(w_sign)))
        print('sanity weights', self.fc1.weight[0].shape, self.fc1.weight.shape)

        if self.sign == 'pospos':    # Q1
            # mask = torch.where(a_pos and w_pos, 1, 0)
            mask = a_pos * w_pos
        elif self.sign == 'posneg':  # Q2
            # mask = torch.where(a_pos and w_neg, 1, 0)
            mask = a_pos * w_neg
        elif self.sign == 'negpos':  # Q3
            # mask = torch.where(a_neg and w_pos, 1, 0)
            mask = a_neg * w_pos
        elif self.sign == 'negneg':  # Q4
            # mask = torch.where(a_neg and w_neg, 1, 0)
            mask = a_neg * w_neg
        elif self.sign == 'pos':
            mask = a_pos * w_pos + a_pos * w_pos  # mutually exclusive
        elif self.sign == 'neg':
            mask = a_pos * w_neg + a_neg * w_pos  # mutually exclusive
        else:
            return x

        print(f"quadrant {self.sign}, {torch.sum(mask, dim=-1)/512}")
        print(f"a_pos {torch.sum(a_pos, dim=-1)}, \
                a_neg {torch.sum(a_neg, dim=-1)}, \
              w_pos {torch.sum(w_pos)}, w_neg {torch.sum(w_neg)}")
        print(f"frac: a_pos {torch.sum(a_pos, dim=-1)/512}, \
                      a_neg {torch.sum(a_neg, dim=-1)/512}, \
              w_pos {torch.sum(w_pos)/512}, w_neg {torch.sum(w_neg)/512}")
        return x * mask

    def forward(self, x):
        ''' mask out all features excluding [fids]'''
        # x = self.norm(x)

        x.register_hook(self.mask_quadrant)
        x = self.mask_quadrant(x)

        if self.sign in ['posneg', 'negpos', 'neg']:
            # flip sign to enable flow of gradient despite neg sign
            # to enable attribution viz (otherwise gradient all 0)
            x = self.fc1(x) * -1.
        else:
            x = self.fc1(x)

        # x = self.fc1(x)
        return x.squeeze(1)

    # def mask_features(self, x, fids=None):
    #     ''' mask fids in calculation of feature attribution
    #         fids: feature_ids, list of ints'''
    #     if self.fids:
    #         fids = self.fids
    #     elif not fids:
    #         # fids = [int(torch.argmax(torch.abs(self.fc1.weight)))]
    #         # print([o for o in enumerate(torch.abs(self.fc1.weight).cpu().detach().numpy())])
    #         # [237, 196, 482, 145, 400, 323, 182, 379, 190, 445]
    #         vec = self.fc1.weight[0].cpu().detach().numpy()
    #         fids = [ix for ix, val in sorted(
    #             enumerate(np.abs(vec)),
    #             key=lambda a: a[1],
    #             reverse=True
    #         )]
    #
    #         print(fids[:10])
    #         fids = fids[-2]
    #     self.fids = fids
    #
    #     # print('feature ids to consider:', fids)
    #     mask = torch.zeros_like(x, dtype=torch.int64)
    #     mask[:, fids] = 1
    #
    #     return x * mask
    #
    # def mask_sign(self, x):
    #     weights = self.fc1.weight[0]
    #     bias = self.fc1.bias
    #     contribs = x @ weights + bias
    #     signs = torch.sign(contribs)
    #
    #     if self.sign == 'pos':
    #         mask = torch.tensor([
    #             torch.where(s == 1, 1, 0) for s in signs],
    #                             dtype=torch.int32, device=x.device)
    #     elif self.sign == 'neg':
    #         mask = torch.tensor([
    #             torch.where(s == -1, 1, 0) for s in signs],
    #                             dtype=torch.int32, device=x.device)
    #     else:
    #         return x
    #
    #     print(torch.mean(torch.tensor(
    #         [m.sum() for m in mask], dtype=torch.float16), axis=0),
    #         f'{self.sign} mean of sums')
    #     mask = mask.unsqueeze(-1)
    #     return mask * x
    #
    # def mask_activation(self, x):
    #     vec = self.fc1.weight[0]  # .cpu().detach().numpy()
    #     signs = torch.sign(vec)
    #
    #     if self.sign == 'pos':
    #         mask = torch.where(signs == 1, 1, 0)
    #         # altsigns = torch.where(signs == 1)
    #     elif self.sign == 'neg':
    #         mask = torch.where(signs == -1, 1, 0)
    #         # altsigns = torch.where(signs == -1)
    #     else:
    #         return x
    #
    #     # altmask = torch.zeros_like(vec, dtype=torch.int8)
    #     # altmask[altsigns] = 1
    #     return x * mask

