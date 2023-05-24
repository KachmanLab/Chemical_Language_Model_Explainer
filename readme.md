# Explainability Techniques for Chemical Language Models
This repository accompanies the paper 'Explainability Techniques for Chemical Language Models' with code to reproduce the results and apply the technique to other self-attention encoder architectures.

The repository is split into the following:
```
src/
	model.py		    - AqueousRegModel, CombiRegModel & <REG> tokenizer
	dataloader.py 		- AqueousSolu & CombiSolu-Exp Dataloaders (SolProp)
	explainer.py 		- Explainability code to attribute atom relevance

scripts/
	train_aqueous.py	- training script for AqueousSolu
	explain_aqueous.py 	- inference script + plots + visualization
	train_combi.py 		- training script for CombiSolu-Exp
	explain_aqueous.py 	- inference script + plots + visualization

nemo_src/
    transformer_attn.py - MegaMolBART source augmented with code to extract attention + grads
    infer.py            - MegaMolBART source to load model
    regex_tokenizer.py  - MegaMolBART source for the tokenzier
```

## Setup:
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
    --env WANDB_API_KEY='' \
    nvcr.io/nvidia/clara/megamolbart_v0.2:0.2.0
```
```
download SolProp datasets from https://zenodo.org/record/5970538
extract AqueousSolu.csv and CombiSolu-Exp.csv into /data
```
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

```
# run scripts with
python scripts/train_aqueous.py
...
python scripts/explain_combi.py
```