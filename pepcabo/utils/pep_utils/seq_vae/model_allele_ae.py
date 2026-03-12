from pathlib import Path
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import AlleleDataModule, AlleleDataset, AA_VOCAB

BATCH_SIZE = 256
ENCODER_LR = 1e-3
DECODER_LR = 1e-3
ENCODER_WARMUP_STEPS = 100
DECODER_WARMUP_STEPS = 100
AGGRESSIVE_STEPS = 5

    
class InfoCNNVAE2(pl.LightningModule):
    def __init__(self,
        dataset: AlleleDataset,
        d_model: int = 64,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
    ):
        super().__init__()
        self.dataset = dataset
        self.vocab_size = len(self.dataset.vocab)

        self.kl_factor    = kl_factor
        self.d_model = d_model
        self.min_posterior_std = min_posterior_std
        self.encoder = nn.Sequential(
            nn.Conv1d(1024, 64, kernel_size=9, padding=1),
            nn.ReLU(),
            nn.Dropout1d(0.15),
            nn.LayerNorm([64, 28]),
            nn.Conv1d(64, 128, kernel_size=9,stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout1d(0.15),
            nn.LayerNorm([128, 11])
        )

        conv_output_size = 128 * (11)
        self.encoder_fc = nn.Linear(conv_output_size, self.d_model*2)
        self.decoder_fc = nn.Linear(self.d_model, conv_output_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=9,stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout1d(0.1),
            nn.LayerNorm([64, 28]),
            nn.ConvTranspose1d(64, self.vocab_size, kernel_size=9, padding=1),
        )
    def sample_prior(self, n):
        return torch.randn(n, 1, self.d_model).to(self.device)

    def sample_posterior(self, mu, sigma, n=None):
        if n is not None:
            mu = mu.unsqueeze(0).expand(n, -1, -1, -1)

        return mu + torch.randn_like(mu) * sigma
        


    def encode(self, embeds):
        embeds = embeds.permute(0, 2, 1) 
        batch,_,length = embeds.shape
        x = self.encoder(embeds)
        x = self.encoder_fc(x.reshape(batch,-1))

        mu = x[..., :self.d_model]
        sigma = F.softplus(x[..., self.d_model:]) + self.min_posterior_std
        return mu, sigma

    def decode(self, z, X_gpu=None):
        
        y = self.decoder_fc(z)
        y = y.reshape((y.shape[0],128,11))
        logits = self.decoder(y)

        return logits

    @torch.no_grad()
    def sample(self, n: int = -1, z: Tensor = None,differentiable: bool = False, return_logits: bool = False):
        model_state = self.training
        self.eval()

        if z is None:
            z = self.sample_prior(n)
        else:
            n = z.shape[0]

        logits = self.decode(z)

        self.train(model_state)

        argmax_indices = logits.argmax(dim=1) 
        samples = F.one_hot(argmax_indices, num_classes=len(AA_VOCAB)).permute(0,2,1).float()
        return samples

    def forward(self, tokens, embeds, max=False):
        mu, sigma = self.encode(embeds)
        if not max:
            z = self.sample_posterior(mu, sigma)
        else:
            z = self.sample_posterior(mu, 0)
        logits = self.decode(z)

        tokens = torch.nn.functional.one_hot(tokens, num_classes=len(AA_VOCAB)).permute(0, 2, 1).float()
        recon_loss_all = F.cross_entropy(logits, tokens, reduction='none')
        recon_loss = recon_loss_all.mean()

        sigma2 = sigma.pow(2)
        kldiv2 = 0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1)
        kldiv = kldiv2.mean() 

        primary_loss = recon_loss
        if self.kl_factor != 0:
            primary_loss = primary_loss + self.kl_factor * kldiv
        loss = primary_loss

        return dict(
            loss=loss, z=z,
            recon_loss=recon_loss,
            recon_loss_all=recon_loss_all,
            kldiv=kldiv,
            kldiv2=kldiv2,
            recon_token_acc=(logits.argmax(dim=1) == tokens.argmax(1)).float().mean(),
            recon_string_acc=(logits.argmax(dim=1) == tokens.argmax(1)).all(dim=1).float().mean(dim=0),
            sigma_mean=sigma.mean(),
            mu=mu
        )


class VAEModule(pl.LightningModule):
    def __init__(self,
        dataset: AlleleDataset,
        d_model: int = 128,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore='dataset')
        self.model = InfoCNNVAE2(dataset=dataset,
                                 d_model=d_model,
                                 kl_factor=kl_factor,
                                 min_posterior_std=min_posterior_std) 
        self.dataset = dataset
        self.lr = ENCODER_LR

    def training_step(self, batch, batch_idx):
        def detach_return(d):
            return {
                k: (v.detach() if k != 'loss' else v)
                for k, v in d.items()
            }
        tokens,embeds,names=batch
        outputs = detach_return(self.model(tokens,embeds))
        for k, v in outputs.items():
            try:
                if k != 'loss':
                    self.log(k, v, on_step=False, on_epoch=True, prog_bar=False, logger=False)
                self.log('train/' + k, v, on_step=True, on_epoch=True, prog_bar='recon' in k, logger=True)
            except:
                pass

        return outputs

    def validation_step(self, batch, batch_idx):
        tokens,embeds,names=batch
        outputs = self.model(tokens,embeds)
        for k, v in outputs.items():
            try:

                self.log('validation/' + k, v, on_step=False, on_epoch=True, prog_bar='recon' in k, logger=True, sync_dist=True)
            except:
                pass

    def collate_fn(self,data):
        # Length of longest seq in batch 
        max_size = max([x.shape[-1] for x in data])
        padded = torch.vstack(
            # Pad with stop token
            [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
        )
        return padded

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
                lr=ENCODER_LR,
                weight_decay = 1e-4
            ),
            dict(
                params=decoder_params,
                lr=DECODER_LR,
                weight_decay = 1e-4
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
    dataset = AlleleDataModule(dataset_path)
    module = VAEModule.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"), dataset=dataset)

    return dict( 
        dataset=dataset,
        module=module,
        model=module.model
    )

def fit():
    root_path = Path("../../../../data")
    train  = root_path / "train_allele.csv"
    val = root_path / "val_allele.csv"

    datamodule = AlleleDataModule(BATCH_SIZE,train,val)
    model = VAEModule(dataset=datamodule.train)

    logger = pl.loggers.WandbLogger(project="Allele_VAE")

    check = ModelCheckpoint(
        every_n_epochs=25,
        save_top_k=-1,
        save_last=True
    )
    trainer = pl.Trainer(
        log_every_n_steps=100,
        strategy="auto",
        logger=logger,
        callbacks=[check, RichProgressBar()],
        gradient_clip_val=1.,
        gradient_clip_algorithm='norm',
        detect_anomaly=False,
        max_epochs=500
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=None)

if __name__ == '__main__':
    fit() 
