

""" adapted from ref: https://arxiv.org/abs/2103.15679 """

import torch
from rdkit.Chem import Draw
from rdkit import Chem
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from omegaconf import OmegaConf

cfg = OmegaConf.load('./params.yaml')
basepath = f"./out/{cfg.task.task}/{cfg.split.split}"
mdir = f"{cfg.model.model}-{cfg.head.head}"


class MolecularSelfAttentionViz():
    """ apply self-attention update rule only """

    def __init__(self, n_layers=6, save_heatmap=False, sign=''):
        self.n_layers = n_layers
        self.save_heatmap = save_heatmap
        self.sign = sign
        self.uid = 0

    def avg_heads(self, attn, grad):
        """ identical, increased readability """
        attn_grad = grad * attn
        attn_grad = attn_grad.clamp(min=0)
        return attn_grad.mean(dim=0)

        # attn = grad.permute(0, 2, 1) * attn.permute(0, 2, 1)
        # return attn.clamp(min=0).mean(dim=0)
        # b, ml, _ml = attn.shape
        # sanity = grad.reshape(-1, ml, ml) * attn.reshape(-1, ml, ml)
        # assert torch.allclose(attn, sanity, 1e-4)

    def agg_relevance(self, attn, grad, ml, token=None):
        # init relevancy matrix as identity
        rel = torch.eye(ml, ml)
        all_rel = []
        # cast to float32 for torch.clamp
        attn, grad = attn.float(), grad.float()

        # loop through each layer
        for layer in range(self.n_layers):
            # calculate avg over heads using attn & gradient
            attn_map = self.avg_heads(
                attn[layer, :, :ml, :ml],
                grad[layer, :, :ml, :ml]
            ).cpu().detach()

            # normalize
            # min, max = torch.min(attn_map), torch.max(attn_map)
            # attn_map = (attn_map - min) / (max - min)

            # apply update rule
            a_bar = torch.matmul(attn_map, rel)
            rel = rel + a_bar

            # optionally save heatmaps
            if self.save_heatmap:
                prefix = f'{self.uid}_{self.sign}'
                save_heat(a_bar, ml, token, prefix=f'{prefix}_a_bar_l{layer}')
                all_rel.append(self.get_weights(rel, ml))

                save_heat(rel - torch.eye(ml), ml, token,
                          prefix=f'{prefix}_rel{layer}')
                for h in range(8):  # n_heads
                    save_heat(attn[layer, h, :ml, :ml].cpu().detach(),
                              ml, token, prefix=f'{prefix}_attn{layer}_h{h}')
                    save_heat(grad[layer, h, :ml, :ml].cpu().detach(),
                              ml, token, prefix=f'{prefix}_grad{layer}_h{h}')
                    gradh = grad[layer, h, :ml, :ml].cpu().detach()
                    attnh = attn[layer, h, :ml, :ml].cpu().detach()
                    attngrad = gradh * attnh
                    save_heat(attngrad.clamp(min=0).cpu().detach(),
                              ml, token,
                              prefix=f'{prefix}_attnxgrad{layer}_h{h}')

        if self.save_heatmap:
            save_heat(rel - torch.eye(ml), ml, token, f'{prefix}_full_rel')
            plot_rel_layers(all_rel, ml, token, f'{prefix}')
            save_heat(torch.eye(ml), ml, token, f'{prefix}_identity')
        return rel

    def get_weights(self, rel, ml):
        """ extract weights from <R> token importance """
        # apply mask and remove I diagonal
        rel = rel[:ml, :ml] - torch.eye(ml, ml)
        # extract row, drop <R> token itself
        return np.array(rel[0, 1:])

        # alternatively extract column
        # return np.array(rel[1:, 0])

    def __call__(self, attn, grad, mask, token=None):
        # get mask length
        ml = sum(mask)
        # aggregate relevance matrix R
        rel = self.agg_relevance(attn, grad, ml, token)
        # keep track of uid viz
        self.uid += 1
        # extract token importance as relevance weights to <R> token
        return self.get_weights(rel, ml)


class ColorMapper():
    def __init__(self, color='green', vmin=None, vmax=None, cmap=None):
        self.color = color
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = ['C', 'c', 'O', 'o', 'N', 'B', 'Br', 'F', 'S', 'Cl', 'P',
            '[P]', 'I', 'n', '[n]', 's', '[s]', '[S]',  '[P+]', '[B]', '[N+]',
            '[O-]', '[#6]', '[#7]', '[C@H]', '[C@]', '[C@@]', '[C@@H]', '[nH]',
            '[NH]', '[NH0]', '[SH0]', '[H]', '[N]', '[N@@]', '[15N]', '[15NH]',
            '[P@]', '[NH0+]', '[NH+]', 
            # logP specific below
            '[13C]', '[CH2]', '[CH2+]', '[3H]', '[13C@@H]', '[13CH2]', '[I-]',
            '[Cl+]', '[CH+]', '[CH]', '[2H]', '[11CH]', '[35Cl]', '[P+]',
            '[NH3+]', '[NH2+]', '[CH2-]', '[C-]', '[S+]', '[13CH3]', '[IH+]',
            '[OH+]', '[S-]', '[PH3+]', '[O]', '[OH2+]', '[C]', '[C+]', '[CH-]',
            '[N-]', '[P-]', '[PH+]', '[13CH]', '[18F]', '[n+]'] 
        self.nonatoms = ['-', '=', '#', '@', '[', ']', '(', ')', ':', '/',
            '\\', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '|']

        if cmap:
            self.cmap = cmap
        else:
            self.cmap = sns.light_palette(color, reverse=False, as_cmap=True)
        # TODO validate vmin, vmax scaling works (_autoscale gets called)
        self.norm = Normalize(self.vmin, self.vmax)

    def filter_atoms(self, weight, token):
        ''' filter out non-atom tokens '''
        return [weight[i] for i, t in enumerate(token) if t in self.atoms]

    def __call__(self, weight, token):
        ''' filter relevance weights for atom-only tokens
            and scale by its own min/max'''
        self.norm = Normalize(self.vmin, self.vmax)
        return self.norm(self.filter_atoms(weight, token))

    def to_rdkit_cmap(self, weight):
        '''helper to map to shape required by RDkit to visualize '''
        return {i: [tuple(self.cmap(w))] for i, w in enumerate(weight)}


def make_legend(colormap=None):
    if colormap:
        mapper = ColorMapper(cmap=colormap)
    else:
        mapper = ColorMapper()
    sm = ScalarMappable(cmap=mapper.cmap)#, norm=None)
    sm.set_array([])  # create a scalar mappable without any data

    # Create an empty figure and add the colorbar to it
    fig, ax = plt.subplots(figsize=(1.5, 8))
    fig.subplots_adjust(bottom=0.2)

    cbar = fig.colorbar(sm, cax=ax, orientation='vertical',
                        ticks=list(np.linspace(0, 1, 11)))
                        # label='Relative importance (a.u.)',

    cbar.set_label('Relative importance (a.u.)',
                   fontsize=18)

    # Save the colorbar as an image file
    plt.tight_layout()
    plt.savefig(f'{basepath}/{mdir}/viz/colorbar_au.png',
        dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()

def make_div_legend():
    coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    mapper = ColorMapper(vmin=-1, vmax=1, cmap=coolwarm)
    norm = Normalize(vmin=-1, vmax=1)
    sm = ScalarMappable(cmap=mapper.cmap, norm=norm)
    sm.set_array([])  # create a scalar mappable without any data

    # Create an empty figure and add the colorbar to it
    fig, ax = plt.subplots(figsize=(1.5, 8))
    fig.subplots_adjust(bottom=0.2)

    cbar = fig.colorbar(sm, cax=ax, orientation='vertical',
                        ticks=list(np.linspace(-1, 1, 11)))
                        # label='Relative importance (a.u.)',

    cbar.set_label('Relative importance (a.u.)',
                   fontsize=18)

    # Save the colorbar as an image file
    plt.tight_layout()
    plt.savefig(f'{basepath}/{mdir}/viz/colorbar_au.png',
        dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()



def save_heat(rel, ml, token, prefix=""):
    rel = torch.flipud(rel[:ml, :ml])
    token = ['<R>'] + token
    cmap = sns.cubehelix_palette(
        start=0.5, rot=-0.75, dark=0.15, light=0.95, reverse=True, as_cmap=True
    )
    rel *= 100 / 0.8  # scale for visibility
    ax = sns.heatmap(
        rel,
        square=True,
        xticklabels=np.array(token),
        yticklabels=np.array(list(reversed(token))),
        cmap=cmap,
        # vmin=0., vmax=1.
    )
    plt.tight_layout()
    plt.savefig(f'{basepath}/{mdir}/viz/{prefix}_heatmap.png')
    plt.clf()

    ax = sns.heatmap(
        rel,
        cmap=cmap,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
        # vmin = 0., vmax = 1.,
    )
    plt.tight_layout()
    plt.savefig(f'{basepath}/{mdir}/viz/{prefix}_heatmap_raw.png',
                bbox_inches='tight', pad_inches=0)
    plt.clf()


def plot_rel_layers(all_rel, ml, token, prefix=''):
    cmap = sns.cubehelix_palette(
        start=0.5, rot=-0.75, dark=0.15, light=0.95, reverse=True, as_cmap=True
    ) or 'viridis'

    all_rel = np.array(all_rel)
    # token = ['<R>'] + token
    token = np.array(token)
    print('plot_rel call', all_rel.shape)

    # v1
    plt.xticks(np.arange(len(token)), labels=token)
    plt.yticks(np.arange(6))
    plt.imshow(all_rel, cmap=cmap)

    plt.tight_layout()
    plt.savefig(f'{basepath}/{mdir}/viz/{prefix}_6R_plot.png',
                bbox_inches='tight', pad_inches=0)
    plt.clf()

    # v2
    sns.heatmap(
        all_rel,
        xticklabels=np.array(token),
        yticklabels=np.array(range(6)),
        cmap=cmap,
        # vmin=0., vmax=1.
    )
    plt.tight_layout()
    plt.savefig(f'{basepath}/{mdir}/viz/{prefix}_6R_heatmap.png',
                bbox_inches='tight', pad_inches=0)
    plt.clf()

def plot_weighted_molecule(atom_colors, smiles, token, label, pred, prefix="", savedir=""):
    atom_colors = atom_colors
    bond_colors = {}
    h_rads = {}  # ?
    h_lw_mult = {}  # ?

    label = f'Experimental: {label:.2f}, predicted: {pred:.2f}\n{smiles}'

    mol = Chem.MolFromSmiles(smiles)
    mol = Draw.PrepareMolForDrawing(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(700, 700)
    d.drawOptions().padding = 0.0

    mismatch = int(mol.GetNumAtoms()) - len(atom_colors.keys())
    if mismatch != 0:
        d.DrawMolecule(mol)
    else:
        d.DrawMoleculeWithHighlights(
            mol, label, atom_colors, bond_colors, h_rads, h_lw_mult, -1
        )

    d.FinishDrawing()

    with open(file=f'{savedir}/{prefix}_MolViz.png', mode='wb') as f:
        f.write(d.GetDrawingText())

    # some plotting issues for 'C@@H' and 'C@H' tokens since
    # another H atom is rendered explicitly.
    # Might break for ultra long SMILES using |c:1:| notation
    # vocab = ft_model.cmapper.atoms + ft_model.cmapper.nonatoms
    # if int(mol.GetNumAtoms()) != len(atom_colors.keys()):
    #     print(f"Warning: {int(mol.GetNumAtoms()) - len(atom_colors.keys())}")
    #     print(f"count mismatch for {smiles}:\
    #          {[t for t in token if t  not in vocab]}")
    #     print(f'{token}')

    # d.DrawMoleculeWithHighlights(
    #     mol, label, atom_colors, bond_colors, h_rads, h_lw_mult, -1
    # )
    # # todo legend
    # d.FinishDrawing()

# TODO
# draw full molecule in the corner
# draw ecfp4 fragments for whole dataset

# draw aggregated ECFP4 fragments, aggregated & weighted by
# regression head weights for each (?top-n) fragments
# https://github.com/rdkit/rdkit/blob/0e7871dc5ed7a690cae9607de0ce866e49a886b4/rdkit/Chem/Draw/__init__.py#L736
#submol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, atomsToUse=atomsToUse))

# TODO OURS
# split regression head backprop into [512] features,
# obtain individual attribution weight

# show sum(per-feature-attrib) == total-feature-attrib


