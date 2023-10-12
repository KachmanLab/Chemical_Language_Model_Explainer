from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps

from rdkit.Chem import Draw

mol = Chem.MolFromSmiles('c1ccccc1CC1CC1')

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


bi = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi)

list_bits = [(mol, x, bi) for x in fp.GetOnBits()]   # (mol, bit, bit info)
legends = [str(x) for x in fp.GetOnBits()]
Draw.DrawMorganBits(list_bits, molsPerRow=4,legends=legends)


