from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
from rdkit.Chem.Draw import IPythonConsole
import json
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from src.dataloader import AqSolDataset
#drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
#drawOptions.prepareMolsBeforeDrawing = False

with open('/workspace/scripts/aqueous_config.json', 'r') as f:
    cfg = json.load(f)

pl.seed_everything(cfg['seed'])
test_dataset = AqSolDataset('/workspace/data/AqueousSolu.csv', 'test', 
    cfg['acc_test'], cfg['split'], data_seed=cfg['seed'])

smiles = test_dataset.smiles
fp_ = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi, nBits=512)



weights_path = None
if weights_path:
    # todo load weights weights.
else:
    weights = np.random.rand(512, 1)

#smi = 'c1ccccc1CC1CC1'
#mol = Chem.MolFromSmiles(smi)

#ECFP (Morgan)
# fpgen = AllChem.GetMorganGenerator(radius=2)
# ao = AllChem.AdditionalOutput()
# ao.CollectBitInfoMap()

#list_bits = [(mol, x, bi) for x in fp.GetOnBits()]   # (mol, bit, bit info)
#legends = [str(x) for x in fp.GetOnBits()]
#Draw.DrawMorganBits(list_bits, molsPerRow=4,legends=legends)

def sort_dict_by_weight(bits_dict, weight_vector, topk=None):
    ''' sort (bit_id: (mol, bit, bit info) dict by regression weight of bit_id)
        args: 
            bits_dict:  dictionary of (bit: (mol, bit, bitinfo))
            weights:    vector of fingerprint length, such as regression fit
        returns:
            bits_dict, sorted descending by bits with highest weight
    '''
    bits_dict = {k: bits_dict.get(k) for k in sorted(
        bits_dict.keys(), key=lambda bid: weight_vector[int(bid)], 
        reverse=True
    )}
    if topk:
        bits_dict = {k: v for i, (k,v) in enumerate(bits_dict.items()) if i < topk}
    return bits_dict 

def get_weights(bits, weight_vector):
    return [f"{weight_vector[int(b)][0]:.3f}\t#{b}" for b in bits]
   # return [f"{b} \t {float_vector[int(b)][0]:.3f}" for b in bits]

def draw_morgan(smi, uid=0):
    mol = Chem.MolFromSmiles(smi)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi, nBits=512)
    bits_dict = {str(x):(mol, x, bi) for x in fp.GetOnBits()}
    bits_dict = sort_dict_by_weight(bits_dict, weights, topk=9)

    #drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    #drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    p = Draw.DrawMorganBits(bits_dict.values(),
                            molsPerRow=3,
                            legends=get_weights(bits_dict.keys(), weights),
                            useSVG=False,
                            #drawOptions=drawOptions,
                            # kwargs={
                            #     #"molSize":(300,300),
                            #     "subImgSize": (300,300)}
                            )
    p.save(f"/workspace/results/aqueous/ecfp/{uid}_morgan_bits.png")
    return p
   

# plot entire test set:
smiles = test_dataset.smiles
for uid, smi in enumerate(smiles):
    plt = draw_morgan(smi, uid=uid)

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


