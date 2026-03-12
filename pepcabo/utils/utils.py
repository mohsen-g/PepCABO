import torch
import math
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import functools
from scipy import stats
import random
import gpytorch
import numpy as np

def lipschitz_loss(z, y, recon_weight):
    if torch.cuda.is_available():
        z = z.cuda()
        y = y.cuda()
    dif_y = (y.unsqueeze(1) - y.unsqueeze(0)).squeeze(-1)
    dif_z = torch.sqrt(torch.sum((z.unsqueeze(1) - z.unsqueeze(0))**2, dim=2) + 1e-10)
    lips = abs(dif_y / (dif_z + 1e-10))
    
    ratio = lips - torch.median(lips)
    ratio = (ratio * (recon_weight * recon_weight[:,None]).pow(0.5))
    ratio = ratio[ratio>0]
    loss = ratio.mean()

    return loss, torch.mean(lips), torch.mean(abs(dif_y)), dif_z.mean()

def contrastive_loss(allele, peptide, affinity_score,temp=1,weight=None):
        peptide = peptide.squeeze(1) 
        if len(allele.size())==3:
            allele = allele.squeeze(1) 

        sim = -torch.norm(peptide - allele, p=2, dim=1)  / temp
        exp_sim = torch.exp(sim)

        mask = affinity_score.unsqueeze(1) >= affinity_score.unsqueeze(0)
        denominator = torch.sum(exp_sim.unsqueeze(0) * mask.float(), dim=1)
        eps = 1e-8
        loss_vec = -sim + torch.log(denominator + eps) 
        if weight is not None:
            weight=weight/(weight.sum())
            return (loss_vec*weight).sum()
        else:
            return loss_vec.mean()


def update_models_end_to_end(
    train_x,
    allele_x,
    train_y_scores,
    objective,
    model,
    mll,
    learning_rte,
    num_update_epochs,
    alpha,
    beta,
    gamma,
    delta,
    zeta= 0,
    tempreture= 1,
    contrastive = False,
):
    '''Finetune VAE end to end with surrogate model
    This method is build to be compatible with the 
    Seq VAE interface
    '''
    objective.p_vae.train()
    objective.a_vae.train()
    model.train()
    
    combined = list(zip(train_x, allele_x,train_y_scores))
    random.shuffle(combined)
    shuffled_list1, shuffled_list2, shuffled_list3 = zip(*combined)

    # Convert back to lists (optional)
    train_x = list(shuffled_list1)
    allele_x = list(shuffled_list2)
    train_y_scores = list(shuffled_list3)

    objective.p_vae.train()
    objective.a_vae.train()
    model.train() 

    optimizer = torch.optim.Adam([
        {'params': objective.p_vae.parameters(), 'lr':learning_rte},
        {'params': model.parameters(), 'lr':0.01 if contrastive else learning_rte}]) #*1e2

    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560/max_string_length)) 
    num_batches = math.ceil(len(train_x) / bsz)

    for batch_idx in range(num_update_epochs):
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
            batch_list = train_x[start_idx:stop_idx]
            allel_batch = allele_x[start_idx:stop_idx]
            z, _, recon_loss, kldiv = objective.vaes_forward(batch_list,allel_batch,contrastive) 

            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float().cuda()

            pred = model(z)
            surr_loss = -mll(pred, batch_y.cuda()) 
          
            data_weighter = DataWeighter()
            batch_y = batch_y.cpu().numpy()
            recon_weight = DataWeighter.normalize_weights(data_weighter.weighting_function(batch_y))
            batch_y = torch.from_numpy(batch_y).cuda()

            recon_weight = torch.from_numpy(recon_weight).cuda()
            recon_loss = (recon_loss*recon_weight).mean()
            
            vae_loss = recon_loss + 0.1 * kldiv      
            lip_loss, lips, dif_y, dif_z = lipschitz_loss(z, batch_y, recon_weight)
            
            dim = z.shape[-1]
            c = math.exp(math.lgamma((dim+1)/2) - math.lgamma(dim/2))*2
            if contrastive:
                con_loss = contrastive_loss(z[:,64:],z[:,:64],batch_y,tempreture)

            loss = alpha * lip_loss + beta * surr_loss + gamma * vae_loss + delta * (dif_z - c).abs()
            if contrastive:
                loss+= zeta*con_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.p_vae.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    objective.p_vae.eval()
    objective.a_vae.eval()
    model.eval()

    return objective, model

def update_surr_model(
    model,
    mll,
    learning_rte,
    train_z,
    train_y,
    n_epochs,
    use_pretrain,
    verbose
):  
    from pepcabo.utils.bo_utils.ppgpr import WeightedVariationalELBO
    #learning_rte/=10
    mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    model = model.train()
    feature_extractor_parametrs = []
    hyper_parameters = []
    gp_parameters = []
    likelihood_params = []
    for i,j in model.named_hyperparameters():
        if not j.requires_grad:
            continue
        elif 'feature_extract' in i :
            feature_extractor_parametrs.append(j)
        elif 'likelihood' in i:
            likelihood_params.append(j)
        else:
            hyper_parameters.append(j)
    for i,j in model.named_variational_parameters():
        if not j.requires_grad:
            continue
        gp_parameters.append(j)

    lr=0.01
    optimizer = torch.optim.Adam([
            {'params': feature_extractor_parametrs , 'lr':lr if use_pretrain else learning_rte }, # 'weight_decay':1e-3
            {'params': hyper_parameters            , 'lr':lr if use_pretrain else learning_rte},
            {'params': gp_parameters               , 'lr':lr if use_pretrain else learning_rte}, #
            {'params': likelihood_params           , 'lr':lr if use_pretrain else learning_rte}]) #


    train_bsz = min(len(train_y),1024)
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda())
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True, drop_last=False)

    for _ in range(n_epochs):
        train_loss = 0.0
        train_loss2 = 0.0
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda()) 
            with torch.no_grad():
                loss2 = -mll2(output, scores)
            #print(-mll2(output, scores))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_loss2 += loss2.item()
    model = model.eval(
        
    )
    return model

class DataWeighter:
    def __init__(self, quantiles=0.95, noises=0.1):
        self.weighting_function = functools.partial(
            DataWeighter.dbas_weights,
            quantile=quantiles,
            noise=noises,
        )

    @staticmethod
    def normalize_weights(weights: np.array):
        """ Normalizes the given weights """
        return weights / np.mean(weights)

    @staticmethod
    def dbas_weights(properties: np.array, quantile: float, noise: float):
        y_star = np.quantile(properties, quantile)
        if np.isclose(noise, 0):
            weights = (properties >= y_star).astype(float)
        else:
            weights = stats.norm.sf(y_star, loc=properties, scale=noise)
        return weights