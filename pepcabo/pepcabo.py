import torch
import gpytorch
import math
import torch.nn.functional as F
import numpy as np
from gpytorch.mlls import PredictiveLogLikelihood
from pepcabo.utils.bo_utils.turbo import TurboState, update_state, generate_batch
from pepcabo.utils.utils import update_models_end_to_end, update_surr_model
from pepcabo.utils.bo_utils.ppgpr import * #2
from pepcabo.utils.pep_utils.seq_vae.data_loader import collate_fn
from pepcabo.utils.pep_utils.load_data import inv as Inversion
from pepcabo.utils.pep_utils.load_data import compute_init_latent
import math


class PepCABOState:

    def __init__(
        self,
        objective,
        train_x,
        train_y,
        train_z,
        train_fo,
        k=50,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=5,
        acq_func='ts',
        verbose=True,
        alpha=10.0, # Lip loss
        beta=1, # Surr loss
        gamma=1, # VAE loss
        delta=1,
        zeta=3,
        tempreture=0.5,
        allele_pscode=None,
        use_pretrain = True,
        path_to_gp_statedict=None,
        force=False,
        guided=False,
        high=False
        ):
        self.objective          = objective         # objective with vae for particular task
        self.train_x            = train_x           # initial train x data
        self.train_y            = train_y           # initial train y data
        self.train_z            = train_z           # initial train z data
        self.train_fo           = train_fo
        self.minimize           = minimize          # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k                  = k                 # track and update on top k scoring points found
        self.num_update_epochs  = num_update_epochs # num epochs update models
        self.init_n_epochs      = init_n_epochs     # num epochs train surr model on initial data
        self.learning_rte       = learning_rte      # lr to use for model updates
        self.bsz                = bsz               # acquisition batch size
        self.tmp_bsz            = bsz
        self.acq_func           = acq_func          # acquisition function: Thompson Sampling (ts)
        self.force              = force
        self.verbose            = verbose
        self.alpha              = alpha
        self.beta               = beta
        self.gamma              = gamma
        self.delta              = delta
        self.zeta               = zeta
        self.tempreture         = tempreture
        self.allele_pscode      = allele_pscode
        self.high = high
        self.path_to_gp_statedict=path_to_gp_statedict
        self.use_pretrain = use_pretrain
        assert acq_func in ["ts"]
        if minimize:
            self.train_y = self.train_y * -1

        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.best_score_seen = torch.max(train_y)
        self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.best_fo = train_fo[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = False # initial training of surrogate model uses all data for more epochs
        self.new_best_found = False

        self.initialize_top_k()
        self.initialize_surrogate_model(self.path_to_gp_statedict)
        self.initialize_tr_state(guided,self.high)
        self.initialize_xs_to_scores_dict()
    

    def initialize_xs_to_scores_dict(self,):
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = [self.train_y.squeeze()[idx].item(),self.train_fo[idx].unsqueeze(-2)]
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict

    def initialize_top_k(self):
        ''' Initialize top k x, y, and zs'''
        self.top_k_scores, top_k_idxs = torch.topk(self.train_y.squeeze(), min(self.k, len(self.train_y)))
        self.top_k_scores = self.top_k_scores.tolist()
        top_k_idxs = top_k_idxs.tolist()
        self.top_k_xs = [self.train_x[i] for i in top_k_idxs]
        self.top_k_zs = self.train_z[top_k_idxs]
        self.top_k_fo = self.train_fo[top_k_idxs]

    def initialize_tr_state(self,guided,high):
        self.tr_state = TurboState(
            dim=self.train_z.shape[-1],
            batch_size=self.bsz, 
            best_value=torch.max(self.train_y).item(),
            failure_tolerance=4 if high else 1,
            length=0.2 if (guided and not high) else 0.8,
            length_max=0.8
            )

        return self
    
    def initialize_surrogate_model(self,path_to_gp_statedict=None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        n_pts = min(self.train_z.shape[0], 1000)
        train_z = self.train_z
        if self.use_pretrain:
            state_dict = torch.load(path_to_gp_statedict)
            hidden_dims = []
            for k, v in state_dict.items():
                if "hidden" in k and "fc.weight" in k:
                    hidden_dims.append(v.shape[0])
            hidden_dims = tuple(hidden_dims)
            try:
                self.model = GPModelDKLExtended(torch.rand(200,128).cuda(),likelihood,hidden_dims=(64,16))
                self.model.load_state_dict(state_dict)
            except:
                self.model = GPModelDKLExtended(torch.rand(1000,128).cuda(),likelihood,hidden_dims=hidden_dims)
                self.model.load_state_dict(state_dict)
            for name,param in self.model.named_parameters():
                if ('variational' in name ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.model.eval()
            self.model.cuda()
            self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=len(train_z))
        else:
            self.model = GPModelDKL(train_z[:n_pts].cuda(),likelihood,dropout_prob=0.05).cuda()
            self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=len(train_z))
            self.model = self.model.eval() 
            self.model = self.model.cuda()

        return self

    def update_next(self, z_next_, y_next_,fo_next_, x_next_, acquisition=False): 
        '''Add new points (z_next, y_next, x_next) to train data
            and update progress (top k scores found so far)
            and update trust region state
        '''
        z_next_ = z_next_.detach().cpu() 
        y_next_ = y_next_.detach().cpu()
        fo_next_= fo_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze() 
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)

        progress = False
        skip_idx = []
        for i, score in enumerate(y_next_):
            if x_next_[i] in self.train_x:
                skip_idx.append(i)
                continue
                
            self.train_x.append(x_next_[i])
            if len(self.top_k_scores) < self.k: 
                self.top_k_scores.append(score.item())
                self.top_k_xs.append(x_next_[i])
                #self.top_k_zs.append(z_next_[i].unsqueeze(-2))
                self.top_k_zs = torch.cat((self.top_k_zs, z_next_[i].unsqueeze(0)))
                self.top_k_fo = torch.cat((self.top_k_fo, fo_next_[i].unsqueeze(0)))
            elif score.item() > min(self.top_k_scores) and (x_next_[i] not in self.top_k_xs):
                # if the score is better than the worst score in the top k list, upate the list
                min_score = min(self.top_k_scores)
                min_idx = self.top_k_scores.index(min_score)
                self.top_k_scores[min_idx] = score.item()
                self.top_k_xs[min_idx] = x_next_[i]
                self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2) # .cuda()
                self.top_k_fo[min_idx] = fo_next_[i]
            #if we imporve
            if score.item() > self.best_score_seen:
                self.progress_fails_since_last_e2e = 0
                progress = True
                self.best_score_seen = score.item() #update best
                self.best_x_seen = x_next_[i]
                self.new_best_found = True
                self.best_fo = fo_next_[i]
        if (not progress) and acquisition: # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        if acquisition:
            self.tr_state = update_state(state=self.tr_state, Y_next=y_next_)
    
        for i in range(len(z_next_)):
            if i not in skip_idx:
                self.train_z = torch.cat((self.train_z, z_next_[i].unsqueeze(0)), dim=-2)
                self.train_y = torch.cat((self.train_y, y_next_[i].unsqueeze(0)), dim=-2)
                self.train_fo = torch.cat((self.train_fo, fo_next_[i].unsqueeze(0)), dim=-2)

        return self

    def update_surrogate_model(self, vanilla): 
        if not self.initial_model_training_complete or vanilla:
            n_epochs = self.init_n_epochs
            train_z = self.train_z
            train_y = self.train_y.squeeze(-1)
        else:
            n_epochs = self.num_update_epochs
            new_zs = self.train_z[-self.tmp_bsz:]
            new_ys = self.train_y[-self.tmp_bsz:].squeeze(-1).tolist()
            train_z = torch.cat((new_zs, self.top_k_zs))
            train_y = torch.tensor(new_ys + self.top_k_scores).float()        

        self.model = update_surr_model(
            self.model,
            self.mll,
            self.learning_rte,
            train_z,
            train_y,
            n_epochs,
            self.use_pretrain,
            verbose = not self.initial_model_training_complete,
        )
        self.initial_model_training_complete = True

        return self

    def update_models_e2e(self):

        '''Finetune VAE end to end with surrogate model'''
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.tmp_bsz:]
        new_ys = self.train_y[-self.tmp_bsz:].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()

        allele = [self.allele_pscode for i in range(len(train_x))]

        self.objective, self.model = update_models_end_to_end(
            train_x,
            allele,
            train_y,
            self.objective,
            self.model,
            self.mll,
            self.learning_rte,
            self.num_update_epochs,
            self.alpha,
            self.beta,
            self.gamma,
            self.delta,
            self.zeta,
            self.tempreture,
            contrastive=self.use_pretrain,
            )
        self.tot_num_e2e_updates += 1

        return self

    def acquisition(self,vanilla=False):
        '''Generate new candidate points, 
        evaluate them, and update data
        '''
        z_next = torch.zeros((0,128))
        y_next = np.zeros((0,))
        fo_next = np.zeros((0,3))
        x_next = []
        y_pred = np.zeros((0,))
        tr_index = None
        original_legth=self.tr_state.length
        while True:
            z_candid, y_candid, tr_index = generate_batch(
                state=self.tr_state,
                model=self.model,
                X=self.top_k_zs,
                Y=torch.tensor(self.top_k_scores),
                batch_size=self.bsz, 
                acqf=self.acq_func,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                tr_index=tr_index,
                vanilla=vanilla
            )
            if True:
                with torch.no_grad():
                    reached_batch=False
                    x_founded=set(self.objective.xs_to_scores_dict.keys())
                    out_dict = self.objective(z_candid,y_candid,force_batch=self.force,batch_size=self.bsz-len(set(x_next)))
                    for i,x in enumerate(out_dict['decoded_xs']):
                        if (not (x in x_founded)):
                            z_next = torch.concat((z_next,out_dict['valid_zs'][i:i+1]),dim=0)
                            y_next = np.concatenate((y_next,out_dict['scores'][i:i+1]),axis=0)
                            fo_next = np.concatenate((fo_next,out_dict['fos'][i:i+1]),axis=0)
                            x_next = x_next+[out_dict['decoded_xs'][i]]
                            y_pred = np.concatenate((y_pred,out_dict['pred'][i:i+1]),axis=0)
                            if len(set(x_next))==self.bsz:
                                reached_batch=True
                                break
                    if (not self.force ) or reached_batch:
                        break
                    self.tr_state.length = min(1.5 * self.tr_state.length, self.tr_state.length_max)
        if self.minimize:
            y_next = y_next * -1

        self.tr_state.length=original_legth
        if len(y_next) != 0:
            y_next = torch.from_numpy(y_next).float()
            fo_next = torch.from_numpy(fo_next).float()
            before=len(self.train_x)
            self.update_next(
                z_next,
                y_next,
                fo_next,
                x_next,
                acquisition=True
            )
            self.tmp_bsz = max(1,len(self.train_x)-before)
        else:
            self.tmp_bsz = 1
            self.progress_fails_since_last_e2e += 1
            self.tr_state = update_state(state=self.tr_state, Y_next=torch.tensor([math.inf if self.minimize else -math.inf]))

    def inversion(self):
        new_xs = self.train_x[-self.tmp_bsz:]
        train_x = new_xs + self.top_k_xs
        bsz=64
        init_z_p=compute_init_latent(self.objective.p_vae,train_x,64,
                                     self.objective.p_vae.dataset.encode,
                                     collate_fn)
        init_z_a=compute_init_latent(self.objective.a_vae,[self.allele_pscode],64,
                                     self.objective.a_vae.dataset.embed,
                                     torch.stack)
        

        final_z_p,model_acc = Inversion(self.objective.p_vae,init_z_p,train_x,64,collate_fn)
        init_z_a=init_z_a.repeat(len(final_z_p),1)        

        final_z=torch.cat((final_z_p,init_z_a.cpu()),dim=1)
        self.train_z[-self.tmp_bsz:] = final_z.reshape(final_z.shape[0], -1)[:self.tmp_bsz].cpu()
        self.top_k_zs = final_z.reshape(final_z.shape[0], -1)[-len(self.top_k_zs):].cpu()
        log_dict = {
                'inv/data':model_acc,
            }
        return self