import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from copy import deepcopy
from typing import List

# from nemo_chem.tokenizer.regex_tokenizer import RegExTokenizer
# from nemo_chem.models.megamolbart.infer import NeMoMegaMolBARTWrapper
from nemo_src.regex_tokenizer import RegExTokenizer
from nemo_src.infer import NeMoMegaMolBARTWrapper
from src.explainer import *

class REGRegExTokenizer(RegExTokenizer):
    def __init__(self):
        super().__init__()
        self.load_tokenizer()
        
        self.reg_token = '<REG>'
        self.vocab[self.reg_token] = 6
        self._update_cache()
        self._compile_regex()

        print(f"mask: {self.mask_id}, sep: {self.sep_id}, reg: {self.reg_id}")

    @property
    def reg_id(self):
        return 6

    def tokenize(self, smis: List[str]):
        tokens = [self.text_to_tokens(s) for s in smis]
        # Prepend <REG> token
        token_ids = [self.token_to_ids(['<REG>']+t) for t in tokens]
        pad_length = max([len(seq) for seq in token_ids])
        
        encoder_masks = [
            ([1] * len(seq)) + ([0] * (pad_length - len(seq))) \
                for seq in token_ids
        ]
        token_ids = [
            seq + ([self.pad_id] * (pad_length - len(seq))) \
                for seq in token_ids 
        ]

        token_ids = torch.tensor(token_ids, dtype=torch.int64).cuda()
        encoder_masks = torch.tensor(encoder_masks,
                                    dtype=torch.int64,
                                    device=token_ids.device)

        return token_ids, encoder_masks

    def tokenize_pair(self, solu_smi: List[str], solv_smi: List[str]):
        solu_t = [self.text_to_tokens(s) for s in solu_smi]
        solv_t = [self.text_to_tokens(s) for s in solv_smi]
        
        # Prepend <REG> token, add <SEP> token between solu and solv
        token_ids = [self.token_to_ids(
            ['<REG>'] + solu + ['<SEP>'] + solv
        ) for solu, solv in zip(solu_t, solv_t)]

        pad_length = max([len(seq) for seq in token_ids])
        encoder_masks = [([1] * len(seq)) + ([0] * (pad_length - len(seq))) \
            for seq in token_ids]
        token_ids = [seq + ([self.pad_id] * (pad_length - len(seq))) \
             for seq in token_ids]

        token_ids = torch.tensor(token_ids, dtype=torch.int64).cuda()
        encoder_masks = torch.tensor(encoder_masks,
                                    dtype=torch.int64,
                                    device=token_ids.device)

        return token_ids, encoder_masks


class RegressionHead(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=[512])
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(1)


class AqueousRegModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.init_molbart()
        self.tokenizer = REGRegExTokenizer()
        self.head = RegressionHead()
        self.explainer = MolecularSelfAttentionViz(save_heatmap=False)
        self.cmapper = ColorMapper()

        self.criterion = nn.HuberLoss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()
        self.learning_rate = 1e-5

    def init_molbart(self):
        molbart_model = NeMoMegaMolBARTWrapper()
        self.mmb = molbart_model.model
        self.mmb.enc_dec_model.enc_dec_model.decoder = None
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr = self.learning_rate, \
            betas=(0.9, 0.999))

    def forward(self, solu_smi):
        """ tokenize SMILES and prepend <REG> token.
            encode using MegaMolBART to obtain latent representation.
            use <REG> token to aggregate into static shape
            apply regression head to obtain logS
        """
        # tokenize smiles string of solute 
        solu_tokens, solu_mask = self.tokenizer.tokenize(solu_smi)
        # encode with MMB
        solu = self.mmb.encode(solu_tokens, solu_mask)
        # apply mask
        solu = solu * solu_mask.unsqueeze(-1)
        # take only the <REG> token
        solu = solu[:, 0]
        # apply regression head and return logS prediction
        return self.head(solu)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        mae = self.criterion_mae(outputs, labels)
        mse = self.criterion_mse(outputs, labels)
        metrics = {
            'loss': loss, 
            'train_mae': mae, 
            'train_mse': mse, 
        }
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        self.mmb.unfreeze()
        with torch.set_grad_enabled(True):
            outputs = self(inputs)
        val_loss = self.criterion(outputs, labels)
        val_mae = self.criterion_mae(outputs, labels)
        val_mse = self.criterion_mse(outputs, labels)
        metrics = {
            'val_loss': val_loss, 
            'val_mae': val_mae, 
            'val_mse': val_mse, 
        }
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        self.mmb.unfreeze()
        with torch.set_grad_enabled(True):
            outputs = self(inputs)
        test_mae = self.criterion_mae(outputs, labels)
        test_mse = self.criterion_mse(outputs, labels)
        metrics = {
            'test_mae': test_mae, 
            'test_mse': test_mse, 
        }
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx):
        """ predict & explain:
            forward with grad enabled to evaluate attn weights and gradients 
            collect attn+grads for each layer, 
            call explainer: propagate relevance, extract <REG> token weights
            call color mapper: map atom-token weights to colors for rdkit plot
        """
        inputs, labels = batch

        with torch.set_grad_enabled(True):
            self.zero_grad()
            preds = self(inputs)
        preds.backward(torch.ones_like(preds))
        
        attn, attn_grads = self.collect_attn_grads() 
        
        # re-construct tokens and masks
        _, masks = self.tokenizer.tokenize(inputs)
        tokens = [self.tokenizer.text_to_tokens(s) for s in inputs]
            
        # extract weights & map colors for all samples in batch:
        rel_weights = [self.explainer(attn[i], attn_grads[i], masks[i], tokens[i]) \
            for i in range(len(inputs))]
        atom_weights = [self.cmapper(rel_weights[i], tokens[i]) \
            for i in range(len(inputs))]
        rdkit_colors = [self.cmapper.to_rdkit_cmap(atom_weights[i]) \
            for i in range(len(inputs))]

        return {"preds": preds, "labels": labels, 
            "smiles": inputs, "tokens": tokens, "masks": masks, 
            "rel_weights": rel_weights, "atom_weights": atom_weights,
            "rdkit_colors": rdkit_colors,
        }

    def collect_attn_grads(self):
        """ collect attention activations (attn) and gradients (attn_grads)
            for each layer.
            returns: attn, attn_grads of shape 
                [batch, n_layers (6), n_heads (8), len_solu_tok, len_solu_tok]
        """
        attn, attn_grads = [], []
        for i in range(6): # self.mmb.num_heads
            m = self.mmb.enc_dec_model.enc_dec_model.encoder.model.layers[i]
            attn.append(
                m.self_attention.core_attention.get_attn()
            )
            attn_grads.append(
                m.self_attention.core_attention.get_attn_gradients()
            )
            
        attn = torch.stack(attn, axis=1)
        attn_grads = torch.stack(attn_grads, axis=1)

        return attn, attn_grads


class CombiRegModel(AqueousRegModel):
    def __init__(self):
        super().__init__()
        self.head.fc1 = nn.Linear(512+1, 64)
        self.head.norm = nn.LayerNorm(normalized_shape=[512+1])

    def forward(self, inputs):
        solu_smi, solv_smi, temperature = inputs

        tokens, mask = self.tokenizer.tokenize_pair(solu_smi, solv_smi)
        pair = self.mmb.encode(tokens, mask)
        pair = pair * mask.unsqueeze(-1)

        pair = pair[:, 0]

        pair = torch.concat([pair, temperature.unsqueeze(1)], 1)
        return self.head(pair)

    def predict_step(self, batch, batch_idx):
        """ predict & explain:
            forward with grad enabled to evaluate attn weights and gradients 
            collect attn+grads for each layer, 
            call explainer: propagate relevance, extract <REG> token weights
            call color mapper: map atom-token weights to colors for rdkit plot
        """
        inputs, labels = batch
        with torch.set_grad_enabled(True):
            self.zero_grad()
            preds = self(inputs)
        preds.backward(torch.ones_like(preds))
        
        # reconstruct tokens & mask
        solu_smi, solv_smi, temperature = inputs
        solu_t = [self.tokenizer.text_to_tokens(s) for s in solu_smi]
        solv_t = [self.tokenizer.text_to_tokens(s) for s in solv_smi]

        # drop <REG> token, replace <SEP> with '.' for rdkit plotting
        tokens = [solu + ['.'] + solv for solu, solv in zip(solu_t, solv_t)]
        _, masks = self.tokenizer.tokenize_pair(solu_smi, solv_smi)

        attn, attn_grads = self.collect_attn_grads() 
        
        # extract weights & map colors for all samples in batch:
        rel_weights = [self.explainer(attn[i], attn_grads[i], masks[i]) \
            for i in range(len(solu_smi))]
        atom_weights = [self.cmapper(rel_weights[i], tokens[i]) \
            for i in range(len(solu_smi))]
        rdkit_colors = [self.cmapper.to_rdkit_cmap(atom_weights[i]) \
            for i in range(len(solu_smi))]
        
        return {"preds": preds, "labels": labels, "tokens": tokens, 
                "solu_smi": solu_smi, "solv_smi": solv_smi, "masks": masks, 
                "rel_weights": rel_weights, "atom_weights": atom_weights,
                "rdkit_colors": rdkit_colors}   


class BaselineAqueousModel(AqueousRegModel):
    def __init__(self):
        """ uses average pooling instead of <REG> token """
        super().__init__()
        self.init_molbart()
        self.head = RegressionHead()
        self.cmapper = ColorMapper()

        self.criterion = nn.HuberLoss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_mae = nn.L1Loss()
        self.learning_rate = 1e-5
    
    def init_molbart(self):
        molbart_model = NeMoMegaMolBARTWrapper()
        self.mmb = molbart_model.model
        self.mmb.enc_dec_model.enc_dec_model.decoder = None
        self.tokenizer = molbart_model.tokenizer
    
    def save_salience(self, grad):
        self.salience = torch.pow(grad, 2)
    
    def get_salience(self):
        return self.salience.detach().cpu()
     
    def forward(self, inputs):
        solu, mask = self._tokenize(inputs)
        solu = self.mmb.encode(solu, mask)
        
        if not self.training:
            solu.register_hook(self.save_salience)
        
        solu = solu * mask.unsqueeze(-1)
        solu = torch.mean(solu, dim=1)
        return self.head(solu)

    def _tokenize(self, smis: List[str]):
        tokens = [self.tokenizer.text_to_tokens(s) for s in smis]
        token_ids = [self.tokenizer.token_to_ids(t) for t in tokens]

        pad_length = max([len(seq) for seq in token_ids])
        encoder_mask = [([1] * len(seq)) + ([0] * (pad_length - len(seq))) for seq in token_ids]
        token_ids = [seq + ([self.tokenizer.pad_id] * (pad_length - len(seq))) for seq in token_ids]

        token_ids = torch.tensor(token_ids, dtype=torch.int64).cuda()
        encoder_mask = torch.tensor(encoder_mask,
                                    dtype=torch.int64,
                                    device=token_ids.device)

        return token_ids, encoder_mask
    
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch

        with torch.set_grad_enabled(True):
            self.zero_grad()
            preds = self(inputs)
        preds.backward(torch.ones_like(preds))
        
        _, masks = self._tokenize(inputs)
        tokens = [self.tokenizer.text_to_tokens(s) for s in inputs]
            
        salience = self.get_salience()
        salience = salience.mean(axis=-1)
        salience_colors = [self.cmapper(salience[i], tokens[i]) \
            for i in range(len(inputs))]

        return {"preds": preds, "labels": labels, 
            "smiles": inputs, "tokens": tokens, "masks": masks, 
            "salience_colors": salience_colors}