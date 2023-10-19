from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
from rdkit.Chem.Draw import IPythonConsole
import json
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
from matplotlib.colors import Normalize

from src.dataloader import AqSolDataset
from src.model import ECFPLinear

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])

smiles = test_dataset.smiles
#fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi, nBits=512)


#try:
load=True
if load:
    subfolders = [f.path for f in os.scandir('/workspace/results/aqueous/models/') \
    if (f.path.endswith('.pt') and f.path.split('/')[-1].startswith('ecfp'))]
    ckpt_path = max(subfolders, key=os.path.getmtime)

    model = ECFPLinear(head=cfg['head']).load_from_checkpoint(ckpt_path)
    weights = model.head.fc1.weight[0].cpu().detach().numpy()
    weights = weights[:, None]
    # TODO ADD torch.ABS() for pos/neg attrib

    print('using trained model weights', ckpt_path)
    print(weights.shape)
    assert weights.shape[0] == 512
    bias = model.head.fc1.bias[0].cpu().detach().numpy()
    print('bias', bias)
    # vec = torch.abs(self.fc1.weight[0]).cpu().detach().numpy()
    # fids = [ix for ix, val in sorted(
    #     enumerate(vec),
    #     key=lambda a: a[1],
    #     reverse=True
    # )]
    # print(fids[:8])

#except:
else:
    weights = np.random.rand(512, 1)
    print('using random weights')

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
        bits_dict = {k: v for i, (k,v) in enumerate(bits_dict.items()) if i < topk}
    return bits_dict 

def get_weights(bits, weight_vector):
    return [f"{weight_vector[int(b)][0]:.3f}\t#{b}" for b in bits]

def make_morgan_dict(smi, nbits=512):
    mol = Chem.MolFromSmiles(smi)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, bitInfo=bi, nBits=nbits
    )
    bits_dict = {str(x):(mol, x, bi) for x in fp.GetOnBits()}
    return bits_dict

def draw_morgan_bits(bits_dict, uid):
    p = Draw.DrawMorganBits(bits_dict.values(),
                            molsPerRow=4,
                            legends=get_weights(bits_dict.keys(), weights),
                            useSVG=False,
                            )
    savedir = "/workspace/results/aqueous/ecfp"
    p.save(f"{savedir}/{uid}_morgan_bits.png")
    return p

def attribute_morgan(smi, bits_dict, weights_vec, norm=True):
    ''' attribute morgan weights to molecule:
        for each on-bit (x), attribute w_x (regression head weight for x)
        to each atom involved in this on-bit '''
    mol = Chem.MolFromSmiles(smi)
    # for atom in mol.GetAtoms():
    #     print("Atom index:", atom.GetIdx(), ", Atomic symbol:", atom.GetSymbol())

    atom_weights = np.zeros(len(mol.GetAtoms()))

    for _mol, x, bi in bits_dict.values():
        assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(_mol)
        w_x = weights_vec[int(x)]
        for _bid, tuples in bi.items():
            atomsToUse = set()
            for (atomId, radius) in tuples:
                atomsToUse.update(extend_morgan(mol, atomId, radius))
                #print(_bit, "\t", atomId, "\t", atomsToUse)
            if norm:
                n = len(atomsToUse)
                atom_weights[list(atomsToUse)] += w_x/n
            else:
                atom_weights[list(atomsToUse)] += w_x
        # NOTE: bi.value tuples are (atom_id_CENTER, radius)
        # for each bit_id entry add attrib


    print(atom_weights) 
    return atom_weights

def extend_morgan(mol, atomId, radius=2):
    ''' get atoms in radius centered on atomId. code from rdkit. '''
    bitPath = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomId)

    # get the atoms for highlighting
    atomsToUse = set((atomId, ))
    for b in bitPath:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())
    return atomsToUse

def to_rdkit_cmap(atom_weight):
    '''helper to map to shape required by RDkit to visualize '''
    cmap = sns.light_palette("green", reverse=False, as_cmap=True)
    atom_weight = Normalize()(atom_weight)
    return {i: [tuple(cmap(w))] for i, w in enumerate(atom_weight)}

def plot_weighted_mol(atom_colors, smiles, logS, pred, uid=""):
    atom_colors = atom_colors
    bond_colors = {}
    h_rads = {} #?
    h_lw_mult = {} 

    label = f'Exp logS: {logS:.2f}, predicted: {pred:.2f}\n{smiles}'

    mol = Chem.MolFromSmiles(smiles)
    mol = Draw.PrepareMolForDrawing(mol)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(700, 700)
    d.drawOptions().padding = 0.0 
    
    d.DrawMoleculeWithHighlights(
        mol, label, atom_colors, bond_colors, h_rads, h_lw_mult, -1
    )
    # todo legend
    d.FinishDrawing()
    
    savedir = "/workspace/results/aqueous/ecfp"
    with open(file=f'{savedir}/{uid}_MorganAttrib.png', mode = 'wb') as f:
        f.write(d.GetDrawingText())
    return d

# plot entire test set:
smiles = test_dataset.smiles
for uid, smi in enumerate(smiles):

    #smi = 'CCCCCCCCCCCCCCO'
    bits_dict = make_morgan_dict(smi, nbits=512)  
    bits_dict = sort_dict_by_weight(bits_dict, weights, topk=16)

    _ = draw_morgan_bits(bits_dict, uid=uid)

    morgan_weights = attribute_morgan(smi, bits_dict, weights)
    morgan_cols = to_rdkit_cmap(morgan_weights)

    logs = 0. 
    pred = 0.
    _ = plot_weighted_mol(morgan_cols, smi, logs, pred, uid)


#https://github.com/rdkit/rdkit/blob/d9d1fe2838053484027ba9f5f74629069c6984dc/rdkit/Chem/Draw/__init__.py#L947

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

#smi = 'c1ccccc1CC1CC1'

#ECFP (Morgan)
# fpgen = AllChem.GetMorganGenerator(radius=2)
# ao = AllChem.AdditionalOutput()
# ao.CollectBitInfoMap()

#list_bits = [(mol, x, bi) for x in fp.GetOnBits()]   # (mol, bit, bit info)
#legends = [str(x) for x in fp.GetOnBits()]
#Draw.DrawMorganBits(list_bits, molsPerRow=4,legends=legends)


### SVG
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

    #drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    # p = Draw.DrawMorganBits(bits_dict.values(), molsPerRow=4, 
    #                     legends=get_weights(bits_dict.keys(), weights))
    #
    # print(type(p))
    # p.save('/workspace/results/aqueous/ecfp/morgan_bits.png')
 

#p.savefig('/workspace/results/aqueous/ecfp/morgan_bits.png',
    #bbox_inches='tight')
                    #drawOptions=drawOptions)
                    #legends=dict_bits.keys())  
#p.save("/workspace/results/aqueous/ecfp/morgan_bits.png")
#with open('/workspace/results/aqueous/ecfp/morgan_bits.svg', 'w') as f:
    #f.write(drawer.GetDrawingText())
    #plt.save(f)
    #f.write(d.GetDrawingText())

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


