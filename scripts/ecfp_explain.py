from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
from rdkit.Chem.Draw import IPythonConsole

#drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
#drawOptions.prepareMolsBeforeDrawing = False


smi = 'c1ccccc1CC1CC1'
mol = Chem.MolFromSmiles(smi)

#ECFP (Morgan)
fpgen = AllChem.GetMorganGenerator(radius=2)
ao = AllChem.AdditionalOutput()
ao.CollectBitInfoMap()

fp = fpgen.GetFingerprint(mol,additionalOutput=ao)
bi = ao.GetBitInfoMap()

svgs_ecfp = []
for bitid in bi:
    mfp2_svg = Draw.DrawMorganBit(mol, bitid, bi, useSVG=True)
    svgs_ecfp.append(mfp2_svg)
    

## RDKIT FP
fpgen = AllChem.GetRDKitFPGenerator()
ao = AllChem.AdditionalOutput()
ao.CollectBitPaths()

fp = fpgen.GetFingerprint(mol,additionalOutput=ao)
rdkbi = ao.GetBitPaths()

svgs_rd = []
for bitid in rdkbi:
    rdk_svg = Draw.DrawRDKitBit(mol, bitid, rdkbi, useSVG=True)
    svgs_rd.append(rdk_svg)


#list_bits = [(mol, x, bi) for x in fp.GetOnBits()]   # (mol, bit, bit info)
#legends = [str(x) for x in fp.GetOnBits()]
#Draw.DrawMorganBits(list_bits, molsPerRow=4,legends=legends)

def sort_dict_by_weight(bits_dict, weight_vector):
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
    return bits_dict 

def get_weights(bits, weight_vector):
    return [f"{weight_vector[int(b)][0]:.3f}\t#{b}" for b in bits]
   # return [f"{b} \t {float_vector[int(b)][0]:.3f}" for b in bits]

weights = np.random.rand(512, 1)

bi = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi, nBits=512)
print(fp)
bits_dict = {str(x):(mol, x, bi) for x in fp.GetOnBits()}
bits_dict = sort_dict_by_weight(bits_dict, weights)
print(bits_dict)
#p = Draw.DrawMorganBits(bits_dict.values(), molsPerRow=4, 
#                            legends=get_weights(bits_dict.keys(), weights),
#                        )
#p.save(f"/workspace/results/aqueous/ecfp/morgan_bits_{smi}.png")

#drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
p = Draw.DrawMorganBits(bits_dict.values(), molsPerRow=4, 
                    legends=get_weights(bits_dict.keys(), weights))

#plt.show()
print(type(p))
p.save('/workspace/results/aqueous/ecfp/morgan_bits.svg')
#p.savefig('/workspace/results/aqueous/ecfp/morgan_bits.png',
    #bbox_inches='tight')
                    #drawOptions=drawOptions)
                    #legends=dict_bits.keys())  
#p.save("/workspace/results/aqueous/ecfp/morgan_bits.png")
#with open('/workspace/results/aqueous/ecfp/morgan_bits.svg', 'w') as f:
    #f.write(drawer.GetDrawingText())
    #plt.save(f)
    #f.write(d.GetDrawingText())

key = list(bits_dict.keys())[0]
tpl = bits_dict.get(key)
print(tpl)
mol, bitId, bitInfo = tpl
atomId = bitInfo[bitId]
submol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, atomsToUse=atomId))


