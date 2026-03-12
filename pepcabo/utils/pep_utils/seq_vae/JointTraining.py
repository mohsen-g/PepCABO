import os
import sys
import argparse
from pathlib import Path

#os.environ["MHCFLURRY_DOWNLOADS_CURRENT_RELEASE"] = "2.0.0"

sys.path.append("../../")

import gpytorch
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy import stats
from torch.optim import Adam

from data_loader import *
from model_allele_ae import InfoCNNVAE2
from model_positional_unbounded import InfoTransformerVAE as Trans
from bo_utils.ppgpr import *

ENCODER_LR = 1e-3
DECODER_LR = 1e-3
TENCODER_LR = 1e-4
TDECODER_LR = 1e-4
AGGRESSIVE_CNN_STEPS = 5
AGGRESSIVE_TRANS_STEPS = 5
GP_START = 50
    
class JointTraining(pl.LightningModule):
    def __init__(self,
        dataset: PairsDataset,
        trans_vae_path: str,
        cnn_vae_path: str,
        temperature: float = 1,
        weighted_loss: bool = True,
        gp_loss: float = 1,
        trans_loss: float = 2,
        cnn_loss: float = 1,
        con_loss: float = 2,
        hidden_size: tuple = (256,256),
        nind=1000,
        ystar=0.65,
        scale=0.15
    ):
        super().__init__()
        self.save_hyperparameters(ignore='dataset')
        self.dataset = dataset
        self.cnn = InfoCNNVAE2(dataset=self.dataset.allele_ds)
        cnn_state_dict = torch.load(cnn_vae_path)
        self.cnn.load_state_dict(cnn_state_dict, strict=True) 
        self.cnn=self.cnn.cuda()
        self.trans = Trans(dataset=self.dataset.peptide_ds,kl_factor=0.1)
        self.nind=nind
        trans_state_dict = torch.load(trans_vae_path)
        self.trans.load_state_dict(trans_state_dict, strict=True) 
        self.ystar=ystar
        self.scale=scale
        self.trans.kl_factor=0.01
        self.cnn.kl_factor=0.01
        self.trans=self.trans.cuda()
        space = torch.randn(self.nind, 128).cuda().float()
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.gp = GPModelDKLExtended(space,self.likelihood,hidden_dims=hidden_size)
        self.mll = WeightedPredictiveLogLikelihood(self.likelihood, self.gp, num_data=len(self.dataset))
        self.normal_mll = gpytorch.mlls.PredictiveLogLikelihood(self.likelihood, self.gp, num_data=len(self.dataset))
        self.temperature = temperature
        self.lr = ENCODER_LR
        self.weighted_loss=weighted_loss
        self.trans_loss = trans_loss
        self.cnn_loss = cnn_loss
        self.con_loss = con_loss
        self.gp_loss = gp_loss


    def contrastive_loss(self, allele, peptide, affinity_score,allel_id,temp,ineq,weight=None):
        peptide = peptide.squeeze(1) 
        if len(allele.size())==3:
            allele = allele.squeeze(1) 
        sim = -torch.norm(peptide - allele, p=2, dim=1)  / temp
        exp_sim = torch.exp(sim)
        mask1 = (ineq==61)
        mask = (
            (affinity_score.unsqueeze(1) > affinity_score.unsqueeze(0)) &
            (allel_id.unsqueeze(1) == allel_id.unsqueeze(0)) &
            (ineq.unsqueeze(0) != 60)
        )
        diag_mask = torch.eye(mask.shape[0], dtype=torch.bool, device=mask.device)
        mask = (mask) | diag_mask
        denominator = torch.sum(exp_sim.unsqueeze(0) * mask.float(), dim=1)[ineq!=62]
        eps = 1e-8
        loss_vec = -sim[ineq!=62] + torch.log(denominator + eps) 
        if weight is not None:
            return (loss_vec*weight[ineq!=62]).mean(),sim,mask1
        else:
            return loss_vec.mean(),sim,mask1

    def get_first_occurrence_indices(self,ids):
        unique_values, inverse_indices = torch.unique(ids, return_inverse=True, sorted=True)

        first_indices = torch.empty_like(ids, dtype=torch.long)

        for i, val in enumerate(unique_values):
            first_indices[i] = (ids == val).nonzero(as_tuple=True)[0][0]

        return first_indices,inverse_indices

    def training_step(self, batch, batch_idx):
        

        peptide_tokens,allele_token,allele_embed,affinity,allele_ids,ineq = batch['pair']        
        batch_y = affinity.cpu().numpy()

        recon_weight = torch.tensor(stats.norm.sf(self.ystar, loc=batch_y, scale=self.scale)).to(self.device)
        recon_weight = torch.clip(recon_weight, 0.1, 4)
        recon_weight = recon_weight/(recon_weight.mean())

        trans_outputs = self.trans(peptide_tokens)
        trans_loss = trans_outputs['loss']
        peptide_z = trans_outputs['z']
        cnn_outputs = self.cnn(allele_token,allele_embed) 

        allele_z = cnn_outputs['z']
        cnn_loss_main = cnn_outputs['loss']

        normal_rnc_loss,sim,mask  = self.contrastive_loss(allele_z,peptide_z,affinity,allele_ids,self.temperature,ineq)
        pearson_corr, pearson_p = stats.pearsonr(affinity[mask].cpu(), sim[mask].detach().cpu().numpy())
        gp_input = torch.concat((peptide_z.squeeze(1),allele_z),dim=-1)
        gp_output = self.gp(gp_input[mask])
        
        gp_loss, weighted_log_likelihood = self.mll(gp_output,affinity[mask],recon_weight[mask])
        gp_loss*=-1

        normal_gp_loss = -self.normal_mll(gp_output,affinity[mask])
        rnc_loss,_,_  = self.contrastive_loss(allele_z,peptide_z,affinity,allele_ids,self.temperature,ineq,recon_weight)
        token,embeds,names=batch['allele']
        cnn_outputs = self.cnn(token,embeds)
        cnn_loss = cnn_outputs['loss']

        if self.weighted_loss:
            total_loss = self.trans_loss*trans_loss+self.cnn_loss*(0.2*cnn_loss+cnn_loss_main)+ self.con_loss*rnc_loss +  self.gp_loss * gp_loss  
        else:
            total_loss = self.trans_loss*trans_loss+self.cnn_loss*(0.2*cnn_loss+cnn_loss_main)+ self.con_loss*normal_rnc_loss+ self.gp_loss * normal_gp_loss
        pred_dist = self.gp.likelihood(gp_output)
        means = pred_dist.mean
        stds = pred_dist.stddev 

        mae = torch.mean(torch.abs(means-affinity[mask]))
        weighted_mae = torch.mean(torch.abs(means-affinity[mask])*recon_weight[mask])

        self.log(f'train_rnc_loss', normal_rnc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_weighted_rnc_loss', rnc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log(f'train_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_gp_loss', normal_gp_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_weighted_gp_loss', gp_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log(f'train_mae_loss', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_weighted_mae_loss', weighted_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_weighted_ll_loss', weighted_log_likelihood, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log(f'corr/train_pearson',pearson_corr,on_step=True, on_epoch=False, prog_bar=True, logger=True)

        progbar_losses = ['loss','recon_loss','kldiv','recon_token_acc','recon_string_acc']
        for name,output_dict in zip(["cnn","trans"],[cnn_outputs,trans_outputs]):
            for k, v in output_dict.items():
                try:
                    self.log(f'train_{name}/{k}', v, on_step=False, on_epoch=True, prog_bar=k in progbar_losses, logger=True)
                except:
                    pass


        return total_loss
    def forward(self, batch):
        
        peptide_tokens,allele_token,affinity,allele_ids,ineq = batch
        batch_size = peptide_tokens.shape[0]
        trans_outputs = self.trans(peptide_tokens)
        peptide_z = trans_outputs['z']
        cnn_outputs = self.cnn(allele_token)
        allele_z = cnn_outputs['z']
        gp_input = torch.concat((peptide_z.squeeze(1),allele_z),dim=-1)
        posterior = self.gp.posterior(gp_input)
        return affinity,posterior.sample()[0,:,0] 
    
    def validation_step(self, batch, batch_idx):
        peptide_token,allele_token,allele_embed,affinity,allele_ids,ineq = batch['pair']
        batch_y = affinity.cpu().numpy()
        
        recon_weight = torch.tensor(stats.norm.sf(self.ystar, loc=batch_y, scale=self.scale)).to(self.device)
        recon_weight = torch.clip(recon_weight, 0.05, 5)
        recon_weight = recon_weight/(recon_weight.mean())

        trans_outputs = self.trans(peptide_token)
        trans_loss = trans_outputs['loss']
        peptide_z = trans_outputs['z']

        cnn_org_outputs = self.cnn(allele_token,allele_embed)
        allele_z = cnn_org_outputs['z']


        rnc_loss,sim,mask  = self.contrastive_loss(allele_z,peptide_z,affinity,allele_ids,self.temperature,ineq,recon_weight)
        pearson_corr, pearson_p = stats.pearsonr(affinity[mask].cpu(), sim[mask].detach().cpu().numpy())
        gp_input = torch.concat((peptide_z.squeeze(1),allele_z),dim=-1)
        gp_output = self.gp(gp_input[mask])
        gp_loss, weighted_log_likelihood = self.mll(gp_output,affinity[mask],recon_weight[mask])
        gp_loss*=-1

        normal_gp_loss = -self.normal_mll(gp_output,affinity[mask])
        normal_rnc_loss,_,_ = self.contrastive_loss(allele_z,peptide_z,affinity,allele_ids,self.temperature,ineq)
        pred_dist = self.gp.likelihood(gp_output)
        means = pred_dist.mean        
        mae = torch.mean(torch.abs(means-affinity[mask]))
        weighted_mae = torch.mean(torch.abs(means-affinity[mask])*recon_weight[mask])

        total_loss = self.trans_loss*trans_loss+self.cnn_loss+self.con_loss*rnc_loss + 0.5* gp_loss
        self.log(f'valid_rnc_loss', normal_rnc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'valid_weighted_rnc_loss', rnc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'valid_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'valid_gp_loss', normal_gp_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'valid_weighted_gp_loss', gp_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'valid_mae_loss', mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'valid_weighted_mae_loss', weighted_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'valid_weighted_ll_loss', weighted_log_likelihood, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'corr/valid_pearson',pearson_corr,on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for name,output_dict in zip(['cnnorg',"trans"],[cnn_org_outputs,trans_outputs]):
            if output_dict is None:
                continue
            for k, v in output_dict.items():
                try:
                    self.log(f'valid_{name}/{k}', v, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                except:
                    pass
    def predict_step(self, batch, batch_idx):
        peptide_token,allele_token,allele_embed,affinity,allele_ids,ineq = batch

        trans_outputs = self.trans(peptide_token)
        peptide_z = trans_outputs['z']
        cnn_org_outputs = self.cnn(allele_token,allele_embed)
        allele_z = cnn_org_outputs['mu']
        _,sim,mask1  = self.contrastive_loss(allele_z,peptide_z,affinity,allele_ids,self.temperature,ineq)
        allele_ids_np = allele_ids[mask1].detach().cpu().numpy()
        affinity_np = affinity[mask1].detach().cpu().numpy()
        sim_np = sim[mask1].detach().cpu().numpy()
        gp_input = torch.concat((peptide_z[mask1].squeeze(1),allele_z[mask1]),dim=-1).detach().cpu().numpy()
        return (gp_input,sim_np,allele_ids_np,affinity_np,mask1)                  
    def configure_optimizers(self):
        cnn_encoder_params = []
        cnn_decoder_params = []
        for name, param in self.cnn.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    cnn_encoder_params.append(param)
                elif 'decoder' in name:
                    cnn_decoder_params.append(param)
                else:
                    raise ValueError(f'Unknown parameter {name}')
        trans_encoder_params = []
        trans_decoder_params = []
        for name, param in self.trans.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    trans_encoder_params.append(param)
                elif 'decoder' in name:
                    trans_decoder_params.append(param)
                else:
                    raise ValueError(f'Unknown parameter {name}')

        feature_extractor_parametrs = []
        hyper_parameters = []
        gp_parameters = []
        likelihood_params = []
        for i,j in self.gp.named_hyperparameters():
            if ('feature_extract' in i):
                feature_extractor_parametrs.append(j)
            elif 'likelihood' in i:
                likelihood_params.append(j)
            else:
                hyper_parameters.append(j)
        for i,j in self.gp.named_variational_parameters():
            gp_parameters.append(j)
        
        def encoder_lr_sched(step):
            return 1

        def decoder_cnn_lr_sched(step):
            if (step + 1) % AGGRESSIVE_CNN_STEPS == 0:
                return 1
            else:
                return 0
        def decoder_trans_lr_sched(step):
            if (step + 1) % AGGRESSIVE_CNN_STEPS == 0:
                return 1
            else:
                return 0

        def gp_ind(step):
            #step-=43*30
            if self.current_epoch<30:
                return 0
            return 1
    
        optimizer = Adam([
            dict(
                params=cnn_encoder_params,
                lr=ENCODER_LR,
                weight_decay=1e-4,
            ),
            dict(
                params=cnn_decoder_params,
                lr=DECODER_LR,
                weight_decay=1e-4,
            ),
            dict(
                params=trans_encoder_params,
                lr=TENCODER_LR
            ),
            dict(
                params=trans_decoder_params,
                lr=TDECODER_LR
            ),
            {'params': feature_extractor_parametrs,'lr': 1e-3, 'weight_decay': 1e-3},
            {'params': hyper_parameters, 'lr': 1e-3},
            {'params': gp_parameters, 'lr': 1e-1},
            {'params': likelihood_params, 'lr': 1e-1},
        ])


        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [encoder_lr_sched, decoder_cnn_lr_sched,encoder_lr_sched, decoder_trans_lr_sched,gp_ind,gp_ind,gp_ind,gp_ind])

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=lr_scheduler,
                interval='step',
                frequency=1
            )
        )
    
def fit():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", default=0.2,type=float)
    parser.add_argument('--weighted-loss',action='store_true')
    parser.add_argument('--trans-loss',default=1,type=float)
    parser.add_argument('--cnn-loss',default=1,type=float)
    parser.add_argument('--con-loss',default=1,type=float)
    parser.add_argument('--gp-loss',default=1,type=float)
    parser.add_argument('--decoder-update',default=20,type=float)
    parser.add_argument("--hidden-size", type=int, nargs="+", default=[256,256])
    parser.add_argument('--nind',type=int,default=1000)
    parser.add_argument('--ystar',type=float,default=0.9)
    parser.add_argument('--scale',type=float,default=0.15)
    parser.add_argument('--batch-size',type=int,default=8192)
    parser.add_argument(
        "--objective",
        type=str,
        default="BA",
        choices=["BA", "PS",'Experimental'],
        help="Initialization strategy for BO candidates"
    )
    args = parser.parse_args()
    Temperature = args.temp
    weighted_loss = args.weighted_loss
    trans_loss = args.trans_loss
    cnn_loss = args.cnn_loss
    con_loss = args.con_loss
    ystar = args.ystar
    scale = args.scale
    nind=args.nind
    gploss=args.gp_loss
    batch_size=args.batch_size
    obj = args.objective
    hidden_size = tuple(args.hidden_size)
    root_path = Path("../../../../data")
    train  = root_path / "train_pairs.csv"
    val = root_path / "val_pairs.csv"

    allele_batch = 512
    allele_train  = root_path / "train_allele.csv"
    allele_val = root_path / "val_allele.csv"
    datamodule = PairsDataModule(batch_size,obj,train,val,
                                 allele_batch,allele_train,allele_val)
    trans_path = root_path / "models/Indp/p_vae.pt"
    cnn_path = root_path / "models/Indp/a_vae.pt"

    model = JointTraining(datamodule.pair_train,trans_path,cnn_path,Temperature,
                          weighted_loss,gploss,trans_loss,cnn_loss,con_loss,hidden_size,
                          nind,ystar,scale)
    logger = pl.loggers.WandbLogger(project="GpJoint")
    check = ModelCheckpoint(
        every_n_epochs=100,
        save_top_k=-1,
        save_last=True
    )

    # Save the top 3 best checkpoints (based on val_loss)
    best_ckpt = ModelCheckpoint(
        monitor="valid_weighted_ll_loss",
        mode="max",                         
        save_top_k=2,                  
        filename="best-{epoch:02d}-{valid_weighted_ll_loss:.4f}",
        verbose=True,
    )
    # Save a checkpoint every 100 epochs
    periodic_ckpt = ModelCheckpoint(
        every_n_epochs=180,         
        save_top_k=-1,             
        filename="epoch-{epoch:04d}",
        save_last=True
    )
    early_stop = EarlyStopping(
        monitor="valid_weighted_ll_loss",  # must match self.log() name
        mode="max",                         # 'min' if lower is better
        patience=30,                       
    )

    logger.log_hyperparams({
        "batch_size": batch_size,
        "train_path": train,
        "val_path": val,
        't-loss': trans_loss,
        'cnn-loss': cnn_loss,
        'con-loss': con_loss,
        'obj': obj
        })
    trainer = pl.Trainer(
        log_every_n_steps=5,
        strategy="auto",
        logger=logger,
        callbacks=[periodic_ckpt,early_stop,best_ckpt],
        gradient_clip_val=1,
        gradient_clip_algorithm='norm',
        max_epochs=500, 
        check_val_every_n_epoch=2,
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    fit() 
