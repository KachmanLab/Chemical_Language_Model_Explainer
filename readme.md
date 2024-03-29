# Explainability Techniques for Chemical Language Models
This repository accompanies the paper 'Explainability Techniques for Chemical Language Models' with code to reproduce the results and apply the technique to other self-attention encoder architectures.
The paper can be found on Arxiv: https://arxiv.org/abs/2305.16192

Fully reproducible runs using dvc & mlflow with model checkpoints and all test set visualizations can be explored at https://dagshub.com/stefanhoedl/Chemical_Language_Model_Explainer.

The repository is split into the following:
```
src/
    model.py            - AqueousRegModel, CombiRegModel & <REG> tokenizer
    dataloader.py       - AqueousSolu & CombiSolu-Exp Dataloaders (SolProp)
    explainer.py        - Explainability code to attribute atom relevance

scripts/
    train_aqueous.py	- training script for AqueousSolu
    explain_aqueous.py 	- inference script + plots + visualization
    predict_aqueous.py  - train+val+test predition + atom weights to .csv
    aqueous_config.json - Aqueous config parameters

    train_combi.py      - training script for CombiSolu-Exp
    explain_combi.py 	- inference script + plots + visualization
    combi_config.json   - Combi config parameters

nemo_src/
    transformer_attn.py - MegaMolBART source augmented with code to extract attention + grads
    infer.py            - MegaMolBART source to load model
    regex_tokenizer.py  - MegaMolBART source for the tokenzier

results/{aqueous, combi}/
    models/             - model checkpoint file produced by training script
    viz/                - visualizations for test set produced by explain script

dvc.yaml                - dvc stages to reproduce results with 'dvc repro {aqueous, combi}'
```
```
# mkdir -p /workspace/data/aq_proc/{random,accurate,scaffold}
# mkdir -p /workspace/aq_prod/{random,accurate,scaffold}/{mmb-hier,mmb-lin,mmb-ft-hier,mmb-ft-lin,mmb-avg,ecfp}/viz
```

# Setup:
### Clone repository, build MegaMolBART docker container
```
git clone https://github.com/KachmanLab/Chemical_Language_Model_Explainer.git
cd Chemical_Language_Model_Explainer
# pull nvidia megamolbart:v0.2 docker container, mount repo (current directory) into /workspace
docker run \
    --gpus all \
    --name mmb \
    -p 8888:8888 \
    --detach \
    --volume $PWD:/workspace \
    nvcr.io/nvidia/clara/megamolbart_v0.2:0.2.0
```

### Download the SolProp datasets
```
download SolProp datasets from https://zenodo.org/record/5970538
extract AqueousSolu.csv and CombiSolu-Exp.csv into /data
```

### Run & attach to docker container
```
# attach shell to container
docker exec -it mmb bash
# change to /workspace directory
cd /workspace
# install requirements
pip install -r requirements.txt
```

### Add code to extract attention scores + gradients
```
# Option 1: overwrite transformer.py with nemo_src/transformer_attn.py
cp /workspace/nemo_src/transformer_attn.py /opt/conda/lib/python3.8/site-packages/nemo/collections/nlp/modules/common/megatron/transformer.py
```
```
# Option 2: Do it yourself (eg. for your own model):
# ADD to class CoreAttention(MegatronModule):
    def save_attn(self, attn):
        self.attn = attn

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn(self):
        return self.attn

    def get_attn_gradients(self):
        return self.attn_gradients

# ADD AFTER SOFTMAX, BEFORE ATTN_DROPOUT
    if not self.training:
        self.save_attn(attention_probs)
        attention_probs.register_hook(self.save_attn_gradients)
```

### Reproduce everything with DVC
```
# run 
dvc repro aqueous
dvc repro combi
```

### or run train + explain scripts individually
```
python scripts/train_aqueous.py
...
python scripts/explain_combi.py
```

### Citation
If you found this repository useful, please cite our paper:
```
@misc{hödl2023explainability,
      title={Explainability Techniques for Chemical Language Models}, 
      author={Stefan Hödl and William Robinson and Yoram Bachrach and Wilhelm Huck and Tal Kachman},
      year={2023},
      eprint={2305.16192},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
