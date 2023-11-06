import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MaskedLinearRegressionHead(pl.LightningModule):
    def __init__(self, fids=None):
        super().__init__()
        # self.norm = nn.LayerNorm(normalized_shape=[512])
        self.fc1 = nn.Linear(512, 1)
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
            vec = self.fc1.weight[0].cpu().detach().numpy()
            fids = [ix for ix, val in sorted(
                enumerate(np.abs(vec)),
                key=lambda a: a[1],
                reverse=True
            )]

            print(fids[:10])
            fids = fids[-2]
        self.fids = fids

        # print('feature ids to consider:', fids)
        mask = torch.zeros_like(x, dtype=torch.int64)
        mask[:, fids] = 1

        return x * mask

    def forward(self, x):
        ''' mask out all features excluding [fids]'''
        x = self.mask_features(x)
        x.register_hook(self.mask_features)

        x = self.fc1(x)
        return x.squeeze(1)


class MaskedRegressionHead(pl.LightningModule):
    def __init__(self, fids=None):
        super().__init__()
        # self.norm = nn.LayerNorm(normalized_shape=[512])
        self.norm = nn.LayerNorm(normalized_shape=[512])
        self.fc1 = nn.Linear(512, 64)
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

        #x = self.mask_features(x)
        x.register_hook(self.mask_features)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)

