from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
# from rdkit.Chem.Draw import SimilarityMaps
# from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
from matplotlib.colors import Normalize
import torch
from src.dataloader import ECFPDataSplit
from src.model import ECFPLinear
from src.explainer import ColorMapper
import hydra
import pickle
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import shap
import json
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR


@hydra.main(
    version_base="1.3", config_path="../conf", config_name="config")
def explain_ecfp(cfg: DictConfig) -> None:

    cfg = OmegaConf.load('./params.yaml')
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.model.seed)
    root = f"./data/{cfg.task.task}/{cfg.split.split}"

    with open(f"{root}/test.pkl", 'rb') as f:
        test = pickle.load(f)
    test_dataset = ECFPDataSplit(test, nbits=cfg.model.nbits)
    # test_loader = DataLoader(test, batch_size=cfg.model.n_batch,
    #                          shuffle=False, num_workers=8)

    basepath = f"./out/{cfg.task.task}/{cfg.split.split}"
    mdir = f"{cfg.model.model}-{cfg.head.head}"
    ckpt_path = f"{basepath}/{mdir}/best.pt"

    if cfg.model.model in 'ecfp':
        if cfg.head.head in ['lin', 'hier']:
            model = ECFPLinear(head=cfg.head.head, dim=cfg.model.nbits)
            model.head.load_state_dict(torch.load(ckpt_path))

            weights = model.head.fc1.weight[0].cpu().detach().numpy()
            weights = weights[:, None]

            print('using trained model weights', ckpt_path)
            print(weights.shape)
            assert weights.shape[0] == cfg.model.nbits

        elif cfg.head.head in ['svr', 'rf']:
            # if cfg.head.head == 'svr':
            #     model = SVR(kernel='rbf')
            # elif cfg.head.head == 'rf':
            #     model = RandomForestRegressor(n_estimators=100,
            #                                   random_state=42)
            with open(ckpt_path, 'rb') as file:
                model = pickle.load(file)

            with open(f"{basepath}/{mdir}/metrics.json", 'r') as f:
                metrics = json.load(f)
            best_fold = np.argmin([v['val_mae'] for k, v in metrics.items()
                                   if k not in ['valid', 'test', 'best_fold']])
            with open(f"{root}/train{best_fold}.pkl", 'rb') as f:
                train = pickle.load(f)
            train_dataset = ECFPDataSplit(train, nbits=cfg.model.nbits)
    else:
        raise NotImplementedError

    # TODO ADD torch.ABS() for pos/neg attrib
    # bias = model.head.fc1.bias[0].cpu().detach().numpy()
    # print('bias', bias)
    # vec = torch.abs(self.fc1.weight[0]).cpu().detach().numpy()
    # fids = [ix for ix, val in sorted(
    #     enumerate(vec),
    #     key=lambda a: a[1],
    #     reverse=True
    # )]
    # print(fids[:8])

    def sort_dict_by_weight(bits_dict, weight_vector, topk=9):
        ''' sort (bit_id: (mol, bit, bit info) dict by regression weight of bit_id)
            args:
                bits_dict:  dictionary of (bit: (mol, bit, bitinfo))
                weights:    vector of fingerprint length, such as regression fit
            returns:
                bits_dict, sorted descending by bits with highest weight
        '''
        bits_dict = {k: bits_dict.get(k) for k in sorted(
            bits_dict.keys(),
            key=lambda id: np.abs(weight_vector)[int(id)],
            reverse=True
        )}
        if topk:
            bits_dict = {k: v for i, (k, v) in enumerate(
                bits_dict.items()) if i < topk}
        return bits_dict

    def get_weights(bits, weight_vector):
        # return [f"{weight_vector[int(b)][0]:.3f}\t#{b}" for b in bits]
        # print([f"{weight_vector[int(b)]:.3f}\t#{b}" for b in bits])
        # print([f"{weight_vector[int(b)][0]:.3f}\t#{b}" for b in bits])
        if cfg.head.head in ['lin']:
            return [f"{weight_vector[int(b)][0]:.3f}\t#{b}" for b in bits]
        elif cfg.head.head in ['svr', 'rf']:
            return [f"{weight_vector[int(b)]:.3f}\t#{b}" for b in bits]

    def calc_shap_weights(explainer, X):
        X = np.array(X)
        if X.shape[0] == cfg.model.nbits:
            X = X[None, ...]

        shap_values = explainer.shap_values(X)[0]
        print('len shap val vec', len(shap_values))
        return shap_values
        # return shap_values
        # print([int(b) for b in bits])
        # print('vals', shap_values)
        # print('len', len(shap_values))
        # print([f"{shap_values[int(b)][0]:.3f}\t#{b}" for b in bits][:7])
        # print("*"*22)
        # [print(f"{shap_values[int(b)]}") for b in bits]
        # return [f"{shap_values[int(b)]}" for b in bits]
        # return [f"{shap_values[int(b)][0]:.3f}\t#{b}" for b in bits]

    def make_morgan_dict(smi, nbits=cfg.model.nbits):
        mol = Chem.MolFromSmiles(smi)
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, bitInfo=bi, nBits=nbits
        )
        bits_dict = {str(x): (mol, x, bi) for x in fp.GetOnBits()}
        return bits_dict

    def draw_morgan_bits(bits_dict, uid):
        p = Draw.DrawMorganBits(bits_dict.values(),
                                molsPerRow=3,
                                legends=get_weights(bits_dict.keys(), weights),
                                useSVG=False,
                                )
        p.save(f"{basepath}/{mdir}/viz/{uid}_morgan_bits.png")
        return p

    def attribute_morgan(smi, bits_dict, weights_vec, norm=True):
        ''' attribute morgan weights to molecule:
            for each on-bit (x), attribute w_x (regression head weight for x)
            to each atom involved in this on-bit '''
        mol = Chem.MolFromSmiles(smi)
        # for atom in mol.GetAtoms():
        #     print("Atom index:", atom.GetIdx(), ", Atomic symbol:", atom.GetSymbol())

        atom_weights = np.zeros(len(mol.GetAtoms()))
        # atom_positive = np.zeros(len(mol.GetAtoms()))
        # atom_negative = np.zeros(len(mol.GetAtoms()))
        for _mol, x, bi in bits_dict.values():
            assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(_mol)
            w_x = weights_vec[int(x)]
            for _bid, tuples in bi.items():
                atomsToUse = set()
                for (atomId, radius) in tuples:
                    # bi.value tuples are (atom_id_CENTER, radius)
                    atomsToUse.update(extend_morgan(mol, atomId, radius))
                    # print(_bit, "\t", atomId, "\t", atomsToUse)

                if norm:
                    w_x = w_x / len(atomsToUse)
                atom_weights[list(atomsToUse)] += w_x

                # if w_x > 0:
                #     atom_positive[list(atomsToUse)] += w_x
                # elif w_x < 0:
                #     atom_negative[list(atomsToUse)] += w_x

        return atom_weights  # , atom_positive, atom_negative

    def extend_morgan(mol, atomId, radius=2):
        ''' get atoms in radius centered on atomId. code from rdkit. '''
        bitPath = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomId)

        # get the atoms for highlighting
        atomsToUse = set((atomId, ))
        for b in bitPath:
            atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
            atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())
        return atomsToUse

    def to_rdkit_cmap(atom_weight, cmap=None):
        '''helper to map to shape required by RDkit to visualize '''
        if not cmap:
            cmap = sns.light_palette("green", reverse=False, as_cmap=True)
        # atom_weight = Normalize()(atom_weight)
        atom_weight = norm(atom_weight)
        return {i: [tuple(cmap(w))] for i, w in enumerate(atom_weight)}

    def plot_weighted_mol(atom_colors, smiles, logS, pred, uid="", suffix=""):
        atom_colors = atom_colors
        bond_colors = {}
        h_rads = {}
        h_lw_mult = {}

        # label = f'Exp logS: {logS:.2f}, predicted: {pred:.2f}\n{smiles}'
        label = ''

        mol = Chem.MolFromSmiles(smiles)
        mol = Draw.PrepareMolForDrawing(mol)
        d = Draw.rdMolDraw2D.MolDraw2DCairo(700, 700)
        d.drawOptions().padding = 0.0

        d.DrawMoleculeWithHighlights(
            mol, label, atom_colors, bond_colors, h_rads, h_lw_mult, -1
        )
        # todo legend
        d.FinishDrawing()

        fname = f'{basepath}/{mdir}/viz/{uid}_MorganAttrib{suffix}.png'
        with open(fname, mode='wb') as f:
            f.write(d.GetDrawingText())
        return d

    # coolwarm = sns.color_palette("coolwarm", as_cmap=True)
    # cmapper = ColorMapper(vmin=-1, vmax=1, cmap=coolwarm)
    # pos_cmapper = ColorMapper(color='blue')
    # neg_cmapper = ColorMapper(color='red')
    div_cmap = sns.color_palette("coolwarm", as_cmap=True)
    mapper = ColorMapper(diverging=True, cmap=div_cmap)
    # pos_cmap = sns.light_palette('red', reverse=False, as_cmap=True)
    # neg_cmap = sns.light_palette('blue', reverse=False, as_cmap=True)

    # plot entire test set:
    smiles = test_dataset.smiles
    labels = test_dataset.labels
    ecfps = test_dataset.ecfp
    # vmin, vmax = 0., 0.
    morgan_preds, morgan_weights = [], []
    # morgan_positive, morgan_negative = [], []

    # background_data = shap.sample(np.array(
    #     valid_dataset.ecfp),
    #     nsamples=100)
    if cfg.head.head == 'svr':
        background_data = shap.kmeans(np.array(train_dataset.ecfp), k=50)
        explainer = shap.KernelExplainer(model.predict,
                                         background_data)
    elif cfg.head.head == 'rf':
        # background_data = shap.sample(np.array(train_dataset.ecfp),
        #                               nsamples=100)
        background_data = np.array(train_dataset.ecfp)
        explainer = shap.TreeExplainer(model,
                                       background_data)
        # all_weights = calc_shap_weights(model, ecfps)

    print('explaining')
    for uid, (smi, logs, ecfp) in enumerate(zip(smiles, labels, ecfps)):
        if cfg.head.head in ['lin', 'hier']:
            pred = model(ecfp[None, ...]).detach().numpy().item()
        elif cfg.head.head in ['svr', 'rf']:
            pred = model.predict(ecfp[None, ...])

        # pred = model.predict(ecfp[None, ...]).detach().numpy().item()
        # pred = model.predict(test_dataset[:64]).detach().numpy().item()

        bits_dict = make_morgan_dict(smi, nbits=cfg.model.nbits)
        # morgan_weight, morgan_pos, morgan_neg = attribute_morgan(
        #     smi, bits_dict, weights)

        if cfg.head.head in ['svr', 'rf']:
            # for svr, rf re-calculate weights using shap,
            # for lin the weights vec stays permanent
            weights = calc_shap_weights(explainer, ecfp)
            [print(f"{weights[int(b)]}", '\t', b) for b in bits_dict]
            # wgt = all_weights[uid]
            # [print(f"{wgt[int(b)]}", '\t', b) for b in bits_dict]
            # assert np.allclose(wgt, weights, 1e-2)

            # weights = model.coef_[0]
            # print(weights.shape)

        morgan_weight = attribute_morgan(smi, bits_dict, weights)

        topk_bits_dict = sort_dict_by_weight(
            bits_dict, weights, topk=cfg.xai.topk)
        _ = draw_morgan_bits(topk_bits_dict, uid=uid)

        morgan_preds.append(pred)
        morgan_weights.append(morgan_weight)
        # morgan_positive.append(morgan_pos)
        # morgan_negative.append(morgan_neg)
        # min, max = np.min(morgan_weights), np.max(morgan_weights)
        # vmin, vmax = np.min(min, vmin), np.max(max, vmax)

        # norm = Normalize(vmin=-7.29, vmax=2.04)
        morgan_div = mapper.to_rdkit_cmap(mapper.div_norm(morgan_weight))
        _ = plot_weighted_mol(morgan_div, smi, logs, pred, uid, '_div')

        norm = Normalize()
        _ = plot_weighted_mol(to_rdkit_cmap(
            morgan_weight, div_cmap), smi, logs, pred, uid, '_reg')
        # norm = Normalize()
        # _ = plot_weighted_mol(to_rdkit_cmap(morgan_pos, pos_cmap), smi, logs, pred, uid, '_pos')
        # norm = Normalize()
        # _ = plot_weighted_mol(to_rdkit_cmap(morgan_neg, neg_cmap), smi, logs, pred, uid, '_neg')

        # if uid > 128:
        #     smiles = smiles[:128]
        #     morgan_weights = morgan_weights[:128]
        #     morgan_preds = morgan_preds[:128]
        #     break

    attributions = pd.DataFrame({
        "smiles": smiles,
        "atom_weights": morgan_weights,
        # "atom_pos": morgan_positive,
        # "atom_neg": morgan_negative,
        "preds": morgan_preds,
        # "labels": list(labels)
        'split': 'test'
    })

    attributions = attributions.reset_index().rename(columns={'index': 'uid'})
    attributions.to_csv(f"{basepath}/{mdir}/attributions.csv", index=False)
# print(vmin, vmax)

# https://github.com/rdkit/rdkit/blob/d9d1fe2838053484027ba9f5f74629069c6984dc/rdkit/Chem/Draw/__init__.py#L947

   #  prOint(type(p))
   #  import io
   #  from PIL import Image, ImageDraw, ImageFont
   #
   #  try:
   #      font = ImageFont.truetype("arial.ttf", 24)  # Specify your font path and size here
   #  except IOError:
   #      font = ImageFont.load_default()
   #  #im = Image.open(io.BytesIO(p))
   #
   #  im = ImageDraw.Draw(p)
   #  text_position = (0, 0)
   #  text_color = (255, 255, 255)
   #  im.text(text_position, "ASRWUADRTUADRURADNW Your Title Here",
   #          #fill=text_color, font=font)
   #          )
   #
   #  print(type(im))
   #  print(type(p))
   #  p.save(f"/workspace/results/aqueous/ecfp/{uid}_morgan_bits.png")
   #  # bytes_container = io.BytesIO()
   #  # im.save(bytes_container, format='PNG')
   #  # image_in_bytes = bytes_container.getvalue()
   #
   #

# smi = 'c1ccccc1CC1CC1'

# ECFP (Morgan)
# fpgen = AllChem.GetMorganGenerator(radius=2)
# ao = AllChem.AdditionalOutput()
# ao.CollectBitInfoMap()

# list_bits = [(mol, x, bi) for x in fp.GetOnBits()]   # (mol, bit, bit info)
# legends = [str(x) for x in fp.GetOnBits()]
# Draw.DrawMorganBits(list_bits, molsPerRow=4,legends=legends)


# SVG
# drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
#     drawOptions.continuousHighlight = False
#     drawOptions.includeMetadata = False
#     drawOptions.prepareMolsBeforeDrawing = False
#     drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    #
    # drawer.FinishDrawing()
    # svg = drawer.GetDrawingText()
    # #plt.savefig(p, f"/workspace/results/aqueous/ecfp/{uid}_morgan_bits.svg")
    # #p.savefig(f"/workspace/results/aqueous/ecfp/{uid}_morgan_bits.svg")
    # #p.save(f"/workspace/results/aqueous/ecfp/{uid}_morgan_bits.svg")
    # with open(f"/workspace/results/aqueous/ecfp/{uid}_morgan_bits.svg", 'w') as f:
    #     f.write(svg)

    # drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    # p = Draw.DrawMorganBits(bits_dict.values(), molsPerRow=4,
    #                     legends=get_weights(bits_dict.keys(), weights))
    #
    # print(type(p))
    # p.save('/workspace/results/aqueous/ecfp/morgan_bits.png')


# p.savefig('/workspace/results/aqueous/ecfp/morgan_bits.png',
    # bbox_inches='tight')
    # drawOptions=drawOptions)
    # legends=dict_bits.keys())
# p.save("/workspace/results/aqueous/ecfp/morgan_bits.png")
# with open('/workspace/results/aqueous/ecfp/morgan_bits.svg', 'w') as f:
    # f.write(drawer.GetDrawingText())
    # plt.save(f)
    # f.write(d.GetDrawingText())

# key = list(bits_dict.keys())[0]
# tpl = bits_dict.get(key)
# print(tpl)
# mol, bitId, bitInfo = tpl
# atomId = bitInfo[bitId]
# submol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, atomsToUse=atomId))

#
# fp = fpgen.GetFingerprint(mol,additionalOutput=ao)
# bi = ao.GetBitInfoMap()
#
# svgs_ecfp = []
# for bitid in bi:
#     mfp2_svg = Draw.DrawMorganBit(mol, bitid, bi, useSVG=True)
#     svgs_ecfp.append(mfp2_svg)
#
#
# ## RDKIT FP
# fpgen = AllChem.GetRDKitFPGenerator()
# ao = AllChem.AdditionalOutput()
# ao.CollectBitPaths()
#
# fp = fpgen.GetFingerprint(mol,additionalOutput=ao)
# rdkbi = ao.GetBitPaths()
#
# svgs_rd = []
# for bitid in rdkbi:
#     rdk_svg = Draw.DrawRDKitBit(mol, bitid, rdkbi, useSVG=True)
#     svgs_rd.append(rdk_svg)

if __name__ == "__main__":
    explain_ecfp()
