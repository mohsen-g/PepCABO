import pandas as pd
import torch
import math
import torch.nn.functional as F
from pepcabo.utils.pep_utils.seq_vae.data_loader import collate_fn
from tqdm import tqdm
import numpy as np
import gpytorch
from pepcabo.utils.bo_utils.turbo import *
import requests
import contextlib
import sys
import os
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
PAD_IDX=26

def load_pairs_train_data(
    obj,
    num_initialization_points,
    predictor,
    xs,
    allele
): 
    if xs is None:    
        raise
    else:
        train_x = np.array(xs)
        alleles = [allele]
    with suppress_stdout():
        results1 = predictor.predict(train_x, alleles)

    results1["affinity"] = 1 - np.log10(results1["affinity"])/(np.log10(50000))
    if obj=='BA':
        train_y = results1["affinity"].values
    elif obj=='PS':
        train_y = results1["presentation_score"].values
    else:
        raise
    train_fo = torch.tensor(results1[['affinity','processing_score','presentation_score']].values).float()
    train_y = torch.from_numpy(train_y).float() 

    train_x = train_x.tolist()[0:num_initialization_points]
    train_y = train_y[0:num_initialization_points]
    train_y = train_y.unsqueeze(-1)
    train_z = None

    assert len(train_x)==len(train_y)
    return train_x, train_z, train_y, train_fo
def initial_peptide_sampling(
    pep_objective,
    init_train_allele,
    gp_path,
    bsz,
    tr_length,
    force
):
    from pepcabo.utils.bo_utils.ppgpr import GPModelDKLExtended
    init_z_a=compute_init_latent(pep_objective.a_vae,init_train_allele,1,
                                 pep_objective.a_vae.dataset.embed,
                                 torch.stack)
    init_z = torch.concat((init_z_a,init_z_a),dim=1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    state_dict = torch.load(gp_path)
    hidden_dims = []
    for k, v in state_dict.items():
        if "hidden" in k and "fc.weight" in k:
            hidden_dims.append(v.shape[0])
    hidden_dims = tuple(hidden_dims)
    gp = GPModelDKLExtended(torch.rand(1000,128).cuda(),likelihood,hidden_dims=hidden_dims).eval().cuda()
    gp.load_state_dict(state_dict)
    
    while True:
        state= TurboState(
            dim=init_z.shape[-1],
            batch_size=bsz, 
            best_value=0,
            failure_tolerance=4,
            length=tr_length
        )
        candidate_z,candidate_y, tr_index = generate_batch(
                state=state,
                model=gp,
                X=init_z,
                Y=torch.tensor([0]),
                batch_size=bsz, 
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        candidate_xs = pep_objective(candidate_z,candidate_y,count_calls=False,batch_size=bsz)
        if (not force) or len(candidate_xs['decoded_xs'])==bsz:
            break
        else:
            tr_length*=1.1

    return candidate_xs

def compute_train_zs(
    pep_objective,
    init_train_peptide,
    init_train_allele,
    bsz=100,
    vanilla=False
):
    init_z_p=compute_init_latent(pep_objective.p_vae,init_train_peptide,bsz,
                                 pep_objective.p_vae.dataset.encode,
                                 collate_fn)
    init_z_a=compute_init_latent(pep_objective.a_vae,init_train_allele,bsz,
                                 pep_objective.a_vae.dataset.embed,
                                 torch.stack)

    init_z_a = init_z_a.repeat(len(init_z_p),1)
    if vanilla:
        final_z=torch.cat((init_z_p.cpu(),init_z_a.cpu()),dim=1)
    else:
        final_z_p, model_acc = inv(pep_objective.p_vae,init_z_p,init_train_peptide,bsz,collate_fn)
        final_z=torch.cat((final_z_p.cpu(),init_z_a.cpu()),dim=1)
    return final_z

def compute_init_latent(model,X,bsz,data_func,collate_f):
    training=False
    if model.training:
        training=True
        model.eval()
    n_batches = math.ceil(len(X)/bsz)
    init_latent = torch.zeros([0])
    if torch.cuda.is_available():
        init_latent = init_latent.cuda()
        
    model.eval() 
    with torch.no_grad():
        for i in range(n_batches):
            xs_batch = X[i*bsz:(i+1)*bsz] 
            s_list = []
            for sample in xs_batch:
                encoded_sample = data_func(sample)
                s_list.append(encoded_sample)
            tokens = collate_f(s_list)
            outtt = model.encode(tokens.cuda())
            try:
                z_sample, _ = outtt
            except:
                z_sample = outtt
            init_latent = torch.cat((init_latent, z_sample), dim=0)
    if training:
        model.train()
    return init_latent

def inv(model,init_latent_o,init_x,bsz,collate_f,argmax_dim=-1):
    #torch.backends.cudnn.enabled = False
    training=False
    if model.training:
        training=True
        model.eval()
    # else:
    #     model.train()
    init_latent = init_latent_o.clone()
    n_batches = math.ceil(len(init_latent)/bsz)
    init_latent.requires_grad_()
    final_z = torch.zeros_like(init_latent)
    model_acc = 0
    for i in range(n_batches):
        optimizer = torch.optim.Adam([
            {'params': init_latent, 'lr': 1e-1},
        ])
        start_idx, stop_idx = i*bsz, (i+1)*bsz
        stop_idx = min(stop_idx,len(init_latent))
        config = init_x[start_idx:stop_idx]
        input_z = init_latent[start_idx:stop_idx]
        if torch.cuda.is_available():
            input_z = input_z.cuda() 
        X_list = []
        for sample in config:
            encoded_x = model.dataset.encode(sample)
            X_list.append(encoded_x)
        X = collate_f(X_list)
        X_gpu = X.clone()
        if torch.cuda.is_available():
            X_gpu = X_gpu.cuda()
        active_indices = torch.arange(stop_idx-start_idx, device=X_gpu.device)  # [0, 1, 2, ..., bsz-1]
        finished_mask = torch.zeros(stop_idx-start_idx, dtype=torch.bool, device=X_gpu.device)  # False=unfinished
        if argmax_dim==-1:
            paded_tokens = torch.full((X_gpu.shape[0], 17), fill_value=26, device=X_gpu.device, dtype=torch.long) 
            paded_tokens[:,:X_gpu.shape[1]]=X_gpu
            paded_tokens[paded_tokens==26]=25
            X_gpu = paded_tokens.clone()
        for e in range(1000):
            if not active_indices.numel():  # Early exit if all finished
                break

            optimizer.zero_grad()
            model.zero_grad()

            logits = model.decode(input_z[active_indices],
                                              X_gpu[active_indices])
            if argmax_dim==-1:
                loss = F.cross_entropy(logits,
                                    X_gpu[active_indices],reduction='none', ignore_index=PAD_IDX)
                loss = loss.mean(dim=-1).mean()#[ X_gpu[active_indices]].mean()
                #print('he')
                active_acc = torch.logical_or(logits.argmax(dim=1) == X_gpu[active_indices],X_gpu[active_indices] == PAD_IDX).float().mean(-1)
            else:
                loss = F.cross_entropy(logits,X_gpu[active_indices])
                active_acc = (logits.argmax(dim=argmax_dim) == X_gpu[active_indices].argmax(dim=argmax_dim)).float().mean(-1)
            
            newly_finished = active_acc == 1.0

            if newly_finished.any():
                global_indices = active_indices[newly_finished]
                if e==0:
                    model_acc += len(global_indices)/len(X_gpu)
                finished_mask[global_indices] = True

                final_z[start_idx+global_indices] = init_latent[start_idx+global_indices].detach()

                active_indices = torch.where(~finished_mask)[0]

            if not active_indices.numel():
                break

            loss.backward()
            optimizer.step()

    if len(active_indices) != 0:
        final_z[start_idx+active_indices] = init_latent[start_idx+active_indices].detach()
    final_z = final_z.reshape(final_z.shape[0], -1)
    if training:
        model.train()
    # else:
        # model.eval()
    return final_z.cpu(),model_acc/n_batches