SolProp v1.2. data collection

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Please cite our work if you wish to use any of the data sets**
**Please contact whgreen@mit.edu for any additional information**

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[General comments]

The SolProp data collection contains thermodynamic properties related to the dissolution of organic solutes in organic solvents and water.
The dataset is limited to neutral solutes and solvents.
Thermodynamic properties included are solvation free energy at 298K (dGsolv), solvation enthalpy at 298K (dHsolv), Abraham Solute Parameters (E,S,A,B,L), 
aqueous solid solubility at 298K (logSaq), and solid solubility in organic solvents at various temperatures (logS)
Some of the data originate from quantum chemical calculations (COSMO-RS theory, with COSMOtherm software), other datasets are experimental curated data from various literature sources.
More details on the construction and curation of each of those datasets can be found in (the supporting information of) the respective publications.
An overview of the different datasets included in this version of the data collection is given below.
A link to the publication and pre-print is provided only if different from the main publication for this version of the data collection: 

'Predicting Solubility Limits of Organic Solutes for a Wide Range of Solvents and Temperatures'
by Florence H. Vermeire, Yunsie Chung, and William H. Green
Department of Chemical Engineering, Massachusetts Institute of Technology
https://doi.org/10.26434/chemrxiv-2022-92hl1

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[Databases overview]

[Data/CombiSolv-QM]

A database for the gas-liquid solvation free energy at 298K, with a reference state of 1 mol/L, calculated with quantum chemistry (COSMO-RS). 
The dataset contains 1 million entries for 284 unique organic solvents (incl water) and 11029 unique organic solutes.
The data is provided as a text file, with solvent and solute represented by their SMILES identifier and InChI identifier and dGsolv at 298K in kcal/mol.

Corresponding pre-print:
https://arxiv.org/pdf/2012.11730.pdf
Corresponding publication:
https://doi.org/10.1016/j.cej.2021.129307

[Data/CombiSolv-Exp]

A database for the gas-liquid solvation free energy at 298K, with a reference state of 1 mol/L, curated from various literature sources (FreeSolv, CompSol, Abraham publications).
The data from the MNSol database is included in the publication, but not in this database because of the proprietary nature of the data.
The dataset contains 10145 entries for 291 unique organic solvents (incl water) and 1368 unique organic solutes.
The data is provided as an excel file, with solvent and solute represented by their SMILES identifier and InChI identifier.
All dGsolv entries from the different databases are given together with the reference (298K, kcal/mol), the value of dGsolv is averaged and the standard deviation is calculated (for specifics see manuscript).

Corresponding pre-print:
https://arxiv.org/pdf/2012.11730.pdf
Corresponding publication:
https://doi.org/10.1016/j.cej.2021.129307

[Data/CombiSolvH-QM]

A database for the gas-liquid solvation enthalpy at 298K, calculated with quantum chemistry (COSMO-RS). 
The dataset contains 800000 entries for 284 unique organic solvents (incl water) and 10891 unique organic solutes.
The data is provided as a text file, with solvent and solute represented by their SMILES identifier and InChI identifier and dHsolv at 298K in kcal/mol.

[Data/CombiSolvH-Exp]

A database for the gas-liquid solvation enthalpy at 298K, curated from various literature sources (CompSol, Acree dataset).
The dataset contains 6322 entries for 1432 unique organic solvents (incl water) and 1665 unique organic solutes.
The data is provided as an excel file, with solvent and solute represented by their SMILES identifier and InChI identifier.
All dHsolv entries from the different databases are given together with the reference (298K, kcal/mol), the value of dHsolv is averaged and the standard deviation is calculated (for specifics see manuscript).

Corresponding pre-print:
https://doi.org/10.33774/chemrxiv-2021-djd3d
Corresponding publication:
https://doi.org/10.1021/acs.jcim.1c01103

[Data/SoluteDB]

A database for empirical Abraham Solute Parameters (E, S, A, B, L) at 298K, from publications and internal databases compiled by Prof. Michael Abraham.
The dataset contains entries for 8366 unique solutes.
The data is provided as an excel file, with the solute represented by its SMILES identifier and InChI identifier.

Corresponding pre-print:
https://doi.org/10.33774/chemrxiv-2021-djd3d
Corresponding publication:
https://doi.org/10.1021/acs.jcim.1c01103

[Data/AqueousSolu]

A database for the aqueous solubility of organic solutes (in water) at 298K, from various literature resources (PHYSPROP, AqSol, ALOGpS, OChem, Drugbank, DLS-100).  
The dataset contains entries for 11804 unique solutes.
The data is provided as a csv file, with solvent and solute represented by their SMILES identifier and InChI identifier.
All logS_aq entries from the different databases are given together with the reference (298K, log(mol/L)), the value of logS_aq is averaged and the standard deviation is calculated (for specifics see manuscript).
Note that for most database entries, more data sources are given compared to the number of data entries. This is because the same data entries appear in multiple sources (usually they are all echoing some original source).
As explained in the manuscript, two data points from different sources are considered different if their value differs by more than 0.11 log(mol/L), i.e. the two reported numbers differ by more than about 25%.

[Data/CombiSolu-Exp]

A database for the solid solubility of organic solutes in organic solvents at various temperatures between 243 - 364 K, from 105 different literature sources.  
The dataset contains 4953 entries for 115 unique solutes and 97 unique solvents. 
The data is provided as a csv file, with solvent and solute represented by their SMILES identifier, InChI identifier, and name. The temperature of the measurement is given together with the units.
The solubility as reported in the source is given together with the standard deviation (if reported), the units and the source.
It should be noted that duplicate entries (same solute, solvent and temperature) in this database are not combined, which is contrary to other databases in this dataset. 
The main reason for this is that the database is used only for testing purpose, while other databases are used for training, validation and testing of machine learning models.

All reported solubilities are converted to molar solubilities (logS [mol/L]) using the solvent density. Note that using the solvent density instead of the density of the solution introduces an error upon conversion.
In most cases the solvent density is determined from interpolation between literature values, if not (e.g. for extrapolation) a warning is given.
Also for the conversion of the reported solubility to the molar solubility (logS [mol/L]) warnings are printed to caution the user, especially if the mole fraction solubility is larger than 0.1. 
If the mole fraction solubility is larger than 0.5 or if the solvent density is not available, the data is not converted to molar solubility.
For more explanation, see the corresponding publication.

[Data/CombiSolu-Exp-HighT]

A database for the solid solubility of solutes in organic solvents at various temperatures between 273 - 593 K, from 68 different literature sources.  
The dataset contains 1306 entries for 67 unique solutes and 15 unique solvents. 
The data is provided as a csv file, with solvent and solute represented by their SMILES identifier, InChI identifier, and name. The temperature of the measurement is given together with the units.
The solubility as reported in the source is given together with the standard deviation (if reported), the units and the source.
It should be noted that duplicate entries (same solute, solvent and temperature) in this database are not combined, which is contrary to other databases in this dataset. 
The main reason for this is that the database is used only for testing purpose, while other databases are used for training, validation and testing of machine learning models.

All reported solubilities are converted to molar solubilities (logS [mol/L]) using the solvent density. Note that using the solvent density instead of the density of the solution introduces an error upon conversion.
In most cases the solvent density is determined from interpolation between literature values, if not (e.g. for extrapolation) a warning is given.
Also for the conversion of the reported solubility to the molar solubility (logS [mol/L]) warnings are printed to caution the user, especially if the mole fraction solubility is larger than 0.1. 
For more explanation, see the corresponding publication.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[Model predictions overview]

[Model_predictions/CombiSolu-Exp-withSolPropML]

The CombiSolu-Exp database with predictions and calculations made by the software SolPropML v1.2. 
(https://github.com/fhvermei/SolProp_ML, https://anaconda.org/fhvermei/solprop_ml).

[Model_predictions/CombiSolu-Exp-HighT-withSolPropML]

The CombiSolu-Exp-HighT database with predictions and calculations made by the software SolPropML v1.2. 
(https://github.com/fhvermei/SolProp_ML, https://anaconda.org/fhvermei/solprop_ml).

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[Machine learning models overview]

[Machine_learning_models/Gsolv]

Machine learning models for solvation free energy at 298K. Models are built with a transfer learning methodology, using both quantum chemistry data as well as experimental data.
More information about model construction and performance can be found in the corresponding publication.
Models are published as part of SolPropML v1.2 (https://anaconda.org/fhvermei/solprop_ml)

Corresponding pre-print:
https://arxiv.org/pdf/2012.11730.pdf
Corresponding publication:
https://doi.org/10.1016/j.cej.2021.129307

[Machine_learning_models/Hsolv]

Machine learning models for solvation enthalpy at 298K. Models are built with a transfer learning methodology, using both quantum chemistry data as well as experimental data.
More information about model construction and performance can be found in the corresponding publication.
Models are published as part of SolPropML v1.2 (https://anaconda.org/fhvermei/solprop_ml)

[Machine_learning_models/Saq]

Machine learning models for aqueous solid solubility at 298K. Models are built using only experimental training data.
More information about model construction and performance can be found in the corresponding publication.
Models are published as part of SolPropML v1.2 (https://anaconda.org/fhvermei/solprop_ml)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
