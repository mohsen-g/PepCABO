import os
import random
import warnings
import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import fire
import mhcflurry
from mhcflurry.testing_utils import startup
import mhcnames

from pepcabo.pepcabo import PepCABOState
from pepcabo.latent_space_objective import LatentSpaceObjective

try:
    import wandb
    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False

warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "True"

startup()

class Optimize(object):
    """
    Run invbo Optimization
    Args:
        obj: Objective function or optimization objective.
        task_id: String id for optimization task
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"pMHC")
        minimize: If True we want to minimize the objective, otherwise we assume we want to maximize the objective
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be ts (ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        update_e2e: If True, we update the models end to end (we run invbo). If False, we never update end to end (we run TuRBO)
        k: We keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc. 
        alpha: Lipschitz regularization weight.
        beta: Surrogate loss weight.
        gamma: Peptide VAE loss weight.
        delta: Pairwise latent-distance regularization weight.
        zeta: Multimodal contrastive alignment weight.
        tempreture: Temperature parameter used in MRNC.
        allele: Target allele for optimization.
        use_pretrain: If True, use pretrained component.
        path_to_gp_statedict: Path to a saved GP state dict used for initialization.
        tr_length: Trust-region length.
        guided: Use guided initialization.
        force: Enforce a fixed number of newly evaluated peptides at each step.
        vanilla: Run the vanilla LSBO.
        high: Use the high-budget variant.
    """
    def __init__(
        self,
        obj,
        task_id: str,
        seed: int=None,
        track_with_wandb: bool=False,
        wandb_entity: str="",
        wandb_project_name: str="",
        minimize: bool=False,
        max_n_oracle_calls: int=500,
        learning_rte: float=0.001,
        acq_func: str="ts",
        bsz: int=5,
        num_initialization_points: int=100,
        init_n_update_epochs: int=20,
        num_update_epochs: int=2,
        e2e_freq: int=10,
        update_e2e: bool=True,
        k: int=50,
        verbose: bool=True,
        alpha : float=10.0,
        beta : int=1,
        gamma : int=1,
        delta : float=1.0,
        zeta  : float=3.0,
        tempreture : float=0.5,
        allele = None,
        use_pretrain= True,
        path_to_gp_statedict=None,
        tr_length = 0.8,
        guided=False,
        force=False,
        vanilla=False,
        high=False
    ):
        assert allele is not None
        self.allele = mhcnames.normalize_allele_name(allele)
        # add all local args to method args dict to be logged by wandb
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.seed = seed
        self.obj=obj
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity 
        self.task_id = task_id
        self.max_n_oracle_calls = max_n_oracle_calls
        self.verbose = verbose
        self.e2e_freq = e2e_freq
        self.update_e2e = update_e2e 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.tempreture = tempreture
        self.num_update_epochs = num_update_epochs
        self.k = k
        self.path_to_gp_statedict = path_to_gp_statedict
        self.tr_length = tr_length
        self.force=force
        self.vanilla=vanilla
        self.high=high
        #print(os.environ["MHCFLURRY_DOWNLOADS_CURRENT_RELEASE"])
        self.predictor = mhcflurry.Class1PresentationPredictor.load()
        self.use_pretrain= use_pretrain
        self.guided=guided

        self.set_seed()
        self.existed_before=False
        peptide_db=pd.read_csv("../data/train_pairs.csv").sample(frac=1).reset_index(drop=True)
        if self.allele in peptide_db.allele.unique().tolist() and (not guided):
            self.existed_before=True
            peptide_db = peptide_db[peptide_db.allele==self.allele].reset_index(drop=True)
            self.num_initialization_points = len(peptide_db)
        else:
            self.num_initialization_points = num_initialization_points

        if wandb_project_name: # if project name specified
            self.wandb_project_name = wandb_project_name
        else: # otherwise use defualt
            self.wandb_project_name = 'pMHC'
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"
        
        
        with open("../data/allele_mapping.pkl", "rb") as f:
            obj = pickle.load(f)
            
            try:
                self.allele_pscode = obj[self.allele]
            except KeyError:
                raise ValueError(f"{self.allele} is not supported")
        self.initialize_objective()
        initial_peptids = self.initial_sampling(peptide_db,self.path_to_gp_statedict,self.guided)
        if initial_peptids is not None:
            self.load_train_data(initial_peptids,self.allele)
            self.init_z()

        
        # initialize latent space objective (self.objective) for particular task
        assert isinstance(self.objective, LatentSpaceObjective), "self.objective must be an instance of LatentSpaceObjective"
        assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"


        # initialize invbo state
        self.invbo_state = PepCABOState(
            objective=self.objective,
            train_x=self.init_train_x,
            train_y=self.init_train_y,
            train_z=self.init_train_z,
            train_fo=self.init_train_fo,
            minimize=minimize,
            k=k,
            num_update_epochs=num_update_epochs,
            init_n_epochs=init_n_update_epochs,
            learning_rte=learning_rte,
            bsz=bsz,
            acq_func=acq_func,
            verbose=verbose,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            zeta=zeta,
            tempreture=tempreture,
            allele_pscode=self.allele_pscode,
            use_pretrain = self.use_pretrain,
            path_to_gp_statedict=self.path_to_gp_statedict,
            force=self.force,
            guided=self.guided,
            high=self.high
        )
        self.log_hist = []
        self.best_found_hist = []
        self.best_found_hist_affinity = []
        self.best_found_hist_processing_score = []
        self.best_found_hist_presentation_score = []
        self.invbo_state.objective.num_calls+=0
        self.invbo_step = 0
    def collate_fn_1d(self,data):
        max_size = max([x.shape[-1] for x in data])
        padded = torch.vstack(
            # Pad with stop token
            [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
        )
        return padded
    
    def initialize_objective(self):
        ''' Initialize Objective for specific task
            must define self.objective object
            '''
        return self
    
    def initial_sampling(self,peptide_db,gp_path,guided):
        ''' Initialize Objective for specific task
            must define self.objective object
            '''
        return self
    
    def init_z(self):
        ''' Initialize Objective for specific task
            must define self.objective object
            '''
        return self


    def load_train_data(self,x=None, allele=None):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_y (a tensor of corresponding latent space points)
        '''
        return self


    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
            torch.backends.cudnn.allow_tf32 = False
        if self.seed is not None:
            torch.manual_seed(self.seed) 
            random.seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(self.seed)

        return self


    def create_wandb_tracker(self):                                         
        name = f'{self.allele}'

        if self.track_with_wandb:
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                name=name,
                config={k: v for method_dict in self.method_args.values() for k, v in method_dict.items()},
            ) 
            self.wandb_run_name = wandb.run.name
            self.wandb_run_id = wandb.run.id                  
        else:
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        
        return self


    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb:
            current_cell = min(self.invbo_state.objective.num_calls,self.max_n_oracle_calls)
            if len(self.log_hist):
                prev_log = self.log_hist[-1]
                prev_cell = prev_log['n_oracle_calls']
                if current_cell==prev_cell:
                    return
                for cell in range(prev_cell+1,current_cell):
                    prev_log['n_oracle_calls']=cell
                    self.tracker.log(prev_log)
                    self.best_found_hist.append(prev_log['best_found'])
                    self.best_found_hist_affinity.append(prev_log['best_fo_aff'])
                    self.best_found_hist_processing_score.append(prev_log['best_fo_processing_score'])
                    self.best_found_hist_presentation_score.append(prev_log['best_fo_presentation_score'])
                    
            
            dict_log = {
                "best_found":self.invbo_state.best_score_seen,
                "n_oracle_calls":current_cell,
                "total_number_of_e2e_updates":self.invbo_state.tot_num_e2e_updates,
                "best_input_seen":self.invbo_state.best_x_seen,
                "invbo_step":self.invbo_step,
                'best_fo_aff':self.invbo_state.best_fo[0],
                'best_fo_processing_score':self.invbo_state.best_fo[1],
                'best_fo_presentation_score':self.invbo_state.best_fo[2],
            }
            dict_log[f"TR_length"] = self.invbo_state.tr_state.length
            self.tracker.log(dict_log)
            self.best_found_hist.append(dict_log['best_found'])
            self.best_found_hist_affinity.append(dict_log['best_fo_aff'])
            self.best_found_hist_processing_score.append(dict_log['best_fo_processing_score'])
            self.best_found_hist_presentation_score.append(dict_log['best_fo_presentation_score'])
            self.log_hist.append(dict_log)
        return self



    def run_invbo(self): 
        ''' Main optimization loop
        '''
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        self.init_log_max=torch.max(self.invbo_state.train_y)
        self.init_log_avg=torch.mean(self.invbo_state.train_y)
        lin_y = 50000**(1-self.invbo_state.train_y)
        self.init_lin_max=torch.min(lin_y)
        self.init_lin_avg=torch.mean(lin_y)
        #main optimization loop
        best = []
        while self.invbo_state.objective.num_calls < self.max_n_oracle_calls:
            self.log_data_to_wandb_on_each_loop()
            if (self.invbo_state.progress_fails_since_last_e2e >= self.e2e_freq) and self.update_e2e and (not self.vanilla) and (self.invbo_step!=0):
                self.invbo_state.update_models_e2e() # Surrogate & VAE update
                self.invbo_state.inversion()

            self.invbo_state.update_surrogate_model(self.vanilla) #self.vanilla
            self.invbo_state.acquisition(self.vanilla)

            if self.invbo_state.tr_state.restart_triggered:
                self.invbo_state.initialize_tr_state(self.guided,self.high)

            if self.invbo_state.new_best_found:
                if self.verbose:
                    print("\nNew best found:")
                    self.print_progress_update()
                self.invbo_state.new_best_found = False
            
            self.invbo_step+=1
            best.append((self.invbo_state.best_score_seen,self.invbo_state.objective.num_calls))
        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        # log top k scores and xs in table
        self.log_data_to_wandb_on_each_loop()
        final_df=self.log_topk_table_wandb()
        return final_df 


    def print_progress_update(self):
        ''' Important data printed each time a new
            best input is found, as well as at the end 
            of the optimization run
            (only used if self.verbose==True)
            More print statements can be added her as desired
        '''
        if self.track_with_wandb:
            print(f"Optimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"Best X Found: {self.invbo_state.best_x_seen}")
        print(f"Best {self.objective.task_id} Score: {self.invbo_state.best_score_seen}")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.invbo_state.objective.num_calls}")

        return self


    def log_topk_table_wandb(self):
        ''' After optimization finishes, log
            top k inputs and scores found
            during optimization '''
        if self.track_with_wandb:
            cols = ["Top K Scores", "Top K Strings"]
            data_list = []
            for ix, score in enumerate(self.invbo_state.top_k_scores):
                data_list.append([ score, str(self.invbo_state.top_k_xs[ix]) ])
            top_k_table = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({f"top_k_table": top_k_table})
            self.best_found_hist=np.array(self.best_found_hist)
            self.lin_scale = np.power(50000,(1-self.best_found_hist))
            self.x = np.linspace(0,1,len(self.best_found_hist))
            self.area = np.trapz(self.best_found_hist, self.x)
            self.lin_area = np.trapz(self.lin_scale, self.x)
            
            dict = {
                "final/linear_best_found":self.lin_scale[-1],
                "final/linear_firstt_found":self.lin_scale[0],
                'final/linear_area':self.lin_area,
                'final/log_best_found':self.best_found_hist[-1],
                'final/aff_best_found':self.best_found_hist_affinity[-1],
                'final/processing_score_best_found':self.best_found_hist_processing_score[-1],
                'final/presentation_score_best_found':self.best_found_hist_presentation_score[-1],
                'final/log_firstt_found':self.best_found_hist[0],
                "final/log_area":self.area,
                'final/anc':0,
            }
            self.tracker.log(dict)
            c_dict = {
                "log_hist":self.best_found_hist,
                "lin_hist":self.lin_scale,
                'affinity_hist':self.best_found_hist_affinity,
                'processing_score_hist':self.best_found_hist_processing_score,
                'presentation_score_hist':self.best_found_hist_presentation_score,
                'seed':self.seed,
                'allele':self.allele,
                'init_log_max':self.init_log_max.item(),
                'init_log_avg':self.init_log_avg.item(),
                'init_lin_min':self.init_lin_max.item(),
                'init_lin_avg':self.init_lin_avg.item(),
            }
            t = [0]*(len(self.best_found_hist)-1)
            t.append(self.invbo_state.train_x)
            c_dict['train_x'] = t
            final_df=pd.DataFrame(c_dict)
            self.tracker.finish()

        return final_df


    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
    os.environ["WANDB_MODE"]="offline"    
    fire.Fire(Optimize)
