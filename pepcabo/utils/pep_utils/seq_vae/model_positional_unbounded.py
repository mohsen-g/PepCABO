import os
import sys
from math import log
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader  import PeptideDataModule, PeptideDataset
BATCH_SIZE = 1024*2
ENCODER_LR = 1e-3
DECODER_LR = 1e-3
ENCODER_WARMUP_STEPS = 100
DECODER_WARMUP_STEPS = 100
AGGRESSIVE_STEPS = 5
PAD_IDX = 26
from Bio.Align import substitution_matrices
Special_TOKENS = ['<start>','<stop>','<PAD>']

def is_valid_peptide(x): return 1 if (len(x)>=8 and len(x)<=15) else 0 

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, dim: int = -1,
                   return_randoms: bool = False, randoms: Tensor = None) -> Tensor:
    """
    Mostly from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    """
    if randoms is None:
        randoms = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1) 
    gumbels = (logits + randoms) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if return_randoms:
        return ret, randoms
    else:
        return ret

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class InfoTransformerVAE(pl.LightningModule):
    def __init__(self,
        dataset: PeptideDataset,
        d_model: int = 64,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 256,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
    ):
        super().__init__()

        self.max_string_length = 256

        self.dataset = dataset
        self.vocab_size = len(self.dataset.vocab)

        self.d_model         = d_model
        self.is_autoencoder  = is_autoencoder
        self.d_decode = d_model

        # TODO
        self.kl_factor    = kl_factor

        self.min_posterior_std = min_posterior_std
        encoder_embedding_dim  = 2 * d_model

        self.encoder_projection =nn.Linear(24,encoder_embedding_dim)
        self.encoder_position_encoding = PositionalEncoding(encoder_embedding_dim, dropout=encoder_dropout, max_len=500)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=encoder_embedding_dim,
            nhead=encoder_nhead,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            activation='relu',
            batch_first=True
        ), num_layers=encoder_num_layers)
        self.encoder_SOS_embedding = torch.nn.Parameter(torch.randn(24))
        self.encoder_EOS_embedding = torch.nn.Parameter(torch.randn(24))
        self.decoder_proj = nn.Linear(64,24*17)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(24, 64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.LayerNorm([64, 17]),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=1,padding=2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.LayerNorm([32, 17]),
            nn.ConvTranspose1d(32, 26, kernel_size=9, stride=1,padding=4),
        )



        blosum62 = substitution_matrices.load("BLOSUM62")
        self.blosum_matrix = torch.tensor(blosum62, dtype=torch.float)
        self.vocab_len = len(list(blosum62.alphabet))+len(Special_TOKENS)
        self.embed_dim = self.blosum_matrix.shape[-1]
        self.embded_matrix = torch.zeros((self.vocab_len,self.embed_dim))
        self.embded_matrix[:self.blosum_matrix.shape[0],:]=self.blosum_matrix
        self.encoder_token_embedding = nn.Embedding.from_pretrained(self.embded_matrix,padding_idx=PAD_IDX,freeze=True)

    def sample_prior(self, n):
        return torch.randn(n, 1, self.d_model).to(self.device)

    def sample_posterior(self, mu, sigma, n=None):
        if n is not None:
            mu = mu.unsqueeze(0).expand(n, -1, -1, -1)

        return mu + torch.randn_like(mu) * sigma

    def generate_pad_mask(self, tokens):
        """ Generate mask that tells encoder to ignore all but first stop token """
        mask = tokens == PAD_IDX
        return mask 
    def generate_lengths(self, tokens):
        return (tokens != PAD_IDX).sum(dim=1)

    def encode(self, tokens, as_probs=False):

        embed = self.encoder_token_embedding(tokens)

        batch_size = embed.size(0)  # 4000

        eos_positions = self.generate_lengths(tokens)-1
        pad_mask = self.generate_pad_mask(tokens)
        embed[:,0] = self.encoder_SOS_embedding
        batch_idx = torch.arange(batch_size, device=embed.device)
        embed[batch_idx, eos_positions, :] = self.encoder_EOS_embedding
        embed = self.encoder_projection(embed)
        embed = self.encoder_position_encoding(embed)
        encoding = self.encoder(embed, src_key_padding_mask=pad_mask)

        mu = encoding[..., :self.d_model]
        sigma = F.softplus(encoding[..., self.d_model:]) + self.min_posterior_std

        mask = (~pad_mask).float()
        mask_expanded = mask.unsqueeze(-1)
        count = mask_expanded.sum(dim=1)

        # Count valid tokens per sequence

        mu_masked = mu * mask_expanded  # zero out padded positions
        sum_mu = mu_masked.sum(dim=1)   # (batch, hidden)
        avg_mu = sum_mu/count

        sigma_masked = sigma * mask_expanded  # zero out padded positions
        sum_sigma= sigma_masked.sum(dim=1)   # (batch, hidden)
        avg_sigma = sum_sigma/count

        return avg_mu, avg_sigma
    
    def decode(self, z, tokens, max_len=16, as_probs=False):
        y = torch.nn.functional.relu(self.decoder_proj(z))
        y = y.reshape((y.shape[0],24,17))
        logits = self.decoder_conv(y)
        return logits
    
    def sample(self, n: int = -1, z: Tensor = None,differentiable: bool = False, return_logits: bool = False, max=False):
        model_state = self.training
        self.eval()
        if z is None:
            z = self.sample_prior(n)
        else:
            n = z.shape[0]

        logits = self.decode(z,None)
        if not max:
            sample, randoms = gumbel_softmax(logits, dim=1, hard=True, return_randoms=True)
            tokens = sample.argmax(dim=1)
        else:
            tokens = logits.argmax(dim=1)

        self.train(model_state)

        if not differentiable:
            sample = tokens

        if return_logits:
            return sample, logits
        else:
            return sample
        
    def is_valid(self, x):
        device = x.device
        x = x.cpu()
        v = [is_valid_peptide(self.dataset.decode(s)) for s in x]

        return torch.tensor(v, dtype=torch.float, device=device) 

    def forward(self, tokens,zs=None):
        if zs is None:
            mu, sigma = self.encode(tokens)

            if self.is_autoencoder:
                z = mu
            else:
                z = self.sample_posterior(mu, sigma)
        else:
            mu = zs.clone()
            z= mu
            sigma = torch.zeros_like(mu)+1
        logits = self.decode(z, tokens)


        paded_tokens = torch.full((tokens.shape[0], 17), fill_value=26, device=self.device, dtype=torch.long) 
        paded_tokens[:,:tokens.shape[1]]=tokens
        paded_tokens[paded_tokens==26]=25
        recon_loss_all = F.cross_entropy(logits, paded_tokens, reduction='none')#, ignore_index=PAD_IDX) #.permute(0, 2, 1)

        recon_loss_all = recon_loss_all.mean(dim=-1)  # Mean per sample (instead of sum / length)
        recon_loss = recon_loss_all.mean()

        # No need for KL divergence when \alpha = 1
        # see https://ojs.aaai.org//index.php/AAAI/article/view/4538 Eq. 6
        # Equation from the original "Auto-Encoding Variational Bayes" paper: https://arxiv.org/pdf/1312.6114.pdf
        sigma2 = sigma.pow(2)
        kldiv2 = 0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1)
        kldiv = kldiv2.mean()  # .sum(dim=(1, 2)).mean(0)

        primary_loss = recon_loss
        if self.kl_factor != 0:
            primary_loss = primary_loss + self.kl_factor * kldiv
        loss = primary_loss

        return dict(
            loss=loss, z=z, mu=mu,
            recon_loss=recon_loss,
            recon_loss_all=recon_loss_all,
            kldiv=kldiv,
            kldiv2=kldiv2,
            recon_token_acc=torch.logical_or(logits.argmax(dim=1) == paded_tokens,paded_tokens == PAD_IDX).float().mean(),
            recon_string_acc=torch.logical_or(logits.argmax(dim=1) == paded_tokens,paded_tokens == PAD_IDX).all(dim=1).float().mean(dim=0),
            sigma_mean=sigma.mean(),
        )


class VAEModule(pl.LightningModule):
    def __init__(self,
        dataset: PeptideDataset,
        d_model: int = 64,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 256,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore='dataset')
        self.kl=kl_factor
        self.maxkl=kl_factor
        self.model = InfoTransformerVAE(dataset=dataset,
                                        d_model=d_model,
                                        is_autoencoder=is_autoencoder,
                                        kl_factor=self.kl,
                                        min_posterior_std=min_posterior_std,
                                        encoder_nhead=encoder_nhead,
                                        encoder_dim_feedforward=encoder_dim_feedforward,
                                        encoder_dropout=encoder_dropout,
                                        encoder_num_layers=encoder_num_layers,
                                        ) 
        self.dataset = dataset
        self.lr = ENCODER_LR
        self.total_step = 97*800

    def cyclical_annealing(self,step, R=0.5, max_kl_weight=1.0):
        if step>=self.total_step:
            return max_kl_weight
        period = self.total_step / 4
        internal_period = step % period
        tau = internal_period / period
        if tau > R:
            return max_kl_weight
        else:
            return min(max_kl_weight, tau / R)
    def training_step(self, batch, batch_idx):
        def detach_return(d):
            return {
                k: (v.detach() if k != 'loss' else v)
                for k, v in d.items()
            }
        current_kl_factor = self.cyclical_annealing(
            step=self.global_step,max_kl_weight=self.maxkl
        )
        self.model.kl_factor = current_kl_factor
        peptide_tokens = batch
        outputs = detach_return(self.model(peptide_tokens))
        self.log("kl_factor", self.model.kl_factor, on_step=True, prog_bar=True)
        for k, v in outputs.items():
            try:
                if k != 'loss':
                    self.log(k, v, on_step=False, on_epoch=True, prog_bar=False, logger=False)
                self.log('train/' + k, v, on_step=True, on_epoch=True, prog_bar='recon' in k, logger=True)
            except:
                pass

        return outputs

    def validation_step(self, batch, batch_idx):
        peptide_tokens = batch
        outputs = self.model(peptide_tokens)

        for k, v in outputs.items():
            try:

                self.log('validation/' + k, v, on_step=False, on_epoch=True, prog_bar='recon' in k, logger=True, sync_dist=True)
            except:
                pass

    
    def on_train_epoch_end(self):
        vv=0
        if self.trainer.is_global_zero:
            with torch.no_grad():
                samples = self.model.sample(100)
                samples = list(map(self.model.dataset.decode,samples))
            if self.current_epoch%10==0 and os.path.isdir(path):
                path = os.path.join(self.trainer.logger.experiment.project,self.trainer.logger.experiment.id, f'samples-epoch={self.current_epoch}.txt')
                with open(path, 'wt') as f:
                    for sample in samples:
                        if len(sample)>=8 and len(sample)<=15:
                            print('VALID:   ', sample, file=f)
                        else:
                            print('INVALID: ', sample, file=f)

    def configure_optimizers(self):
        encoder_params = []
        decoder_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    encoder_params.append(param)
                elif 'decoder' in name:
                    decoder_params.append(param)
                else:
                    raise ValueError(f'Unknown parameter {name}')

        def encoder_lr_sched(step):
            # Use Linear warmup
            return min(step / ENCODER_WARMUP_STEPS, 1.)

        def decoder_lr_sched(step):
            if step < ENCODER_WARMUP_STEPS:
                return 0.
            else:
                if (step - ENCODER_WARMUP_STEPS + 1) % AGGRESSIVE_STEPS == 0:
                    return min((step - ENCODER_WARMUP_STEPS) / (DECODER_WARMUP_STEPS * AGGRESSIVE_STEPS), 1.)
                else:
                    return 0.

        optimizer = Adam([
            dict(
                params=encoder_params,
                lr=ENCODER_LR
            ),
            dict(
                params=decoder_params,
                lr=DECODER_LR
            )
        ])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [encoder_lr_sched, decoder_lr_sched])

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=lr_scheduler,
                interval='step',
                frequency=1
            )
        )

def load(dataset_path, checkpoint_path):
    dataset = PeptideDataModule(dataset_path)
    module = VAEModule.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"), dataset=dataset)

    return dict( 
        dataset=dataset,
        module=module,
        model=module.model
    )

def fit():
    kl = 0.1
    root_path = Path("../../../../data")
    train  = root_path / "train_pairs.csv"
    val = root_path / "val_pairs.csv"
    datamodule = PeptideDataModule(BATCH_SIZE,train,val)
    model = VAEModule(dataset=datamodule.train,kl_factor=kl)

    logger = pl.loggers.WandbLogger(project="Peptide_VAE")

    check = ModelCheckpoint(
        every_n_epochs=25,
        save_top_k=-1,
        save_last=True
    )
    trainer = pl.Trainer(
        log_every_n_steps=100,
        strategy="auto",
        logger=logger,
        callbacks=[check],
        gradient_clip_val=1.,
        gradient_clip_algorithm='norm',
        detect_anomaly=False,
        max_epochs=1000,
        check_val_every_n_epoch=5
    )
    summary = ModelSummary(model, max_depth=2)
    print(summary)
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    fit() 
