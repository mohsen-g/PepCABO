import os
#os.environ["MHCFLURRY_DOWNLOADS_CURRENT_RELEASE"]='2.0.0'
import argparse
import argparse
import torch
from scripts import peptide_optimization


if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(0)
import argparse


ALPHA = 1000  # Lipschitz regularization weight (L_Lip) in end-to-end BO training
DELTA = 1     # Pairwise latent-distance regularization weight (L_z)
BETA  = 1     # Surrogate loss weight
GAMMA = 0.5     # Peptide VAE loss weight (reconstruction + KL)
ZETA  = 3     # Multimodal contrastive alignment weight (L_MRNC)
TRACK_WITH_WANDB = True

parser = argparse.ArgumentParser(
    description="PepCABO: Peptide–MHC binding optimization using latent-space Bayesian Optimization"
)

parser.add_argument(
    "--allele",
    type=str,
    default="HLA-A*25:56",
    help="Target HLA allele for peptide optimization."
)

parser.add_argument(
    "--seed",
    type=int,
    default=1001,
    help="Random seed for reproducibility."
)

parser.add_argument(
    "--no-pretrain",
    action="store_true",
    help="Disable pretrained peptide/allele VAE initialization and train from scratch."
)

parser.add_argument(
    "--init-epoch",
    type=int,
    default=20,
    help="Number of BO initialization epochs before active optimization begins."
)

parser.add_argument(
    "--bsz",
    type=int,
    default=5,
    help="Batch size for BO candidate evaluation."
)

parser.add_argument(
    "--model-path",
    type=str,
    help="Path to pretrained PepCABO model checkpoint."
)


parser.add_argument(
    "--init_strategy",
    type=str,
    default="guided",
    choices=["guided", "random"],
    help="Strategy for generating initial BO candidates (guided uses pretrained latent model)."
)

parser.add_argument(
    "--objective",
    type=str,
    default="BA",
    choices=["BA", "PS"],
    help="Optimization objective: BA (binding affinity) or PS (presentation score)."
)

parser.add_argument(
    "--force",
    action="store_true",
    help="Force the number of newly evaluated peptides at each step to be equal to the batch size."
)

parser.add_argument(
    "--vanilla",
    action="store_true",
    help="Run baseline BO without PepCABO alignment loss."
)

parser.add_argument(
    "--high",
    action="store_true",
    help="Use high oracle-budget setting in manuscript for BO experiments, otherwise low budget"
)

parser.add_argument(
    "--output", 
    type=str, 
    default=None, 
    help="Save output df to this pickle path."
)

parser.add_argument(
    "--task", 
    type=str, 
    default=''
)

args = parser.parse_args()



allele = args.allele
seed = args.seed
bsz = args.bsz
init_epoch = args.init_epoch
model = args.model_path

use_pretrain = not args.no_pretrain
guided = args.init_strategy == "guided"

force = args.force
vanilla = args.vanilla
high = args.high

obj = args.objective
task_id = args.task
if model is not None:
    path_to_peptide_vae_statedict ="f{model}/p_vae.pt"
    path_to_allele_vae_statedict = "f{model}/a_vae.pt"
    if use_pretrain:
        path_to_gp_statedict="f{model}/gp.pt"
    else:
        path_to_gp_statedict=None
else:
    if not use_pretrain:
        path_to_peptide_vae_statedict ="../data/models/Indp/p_vae.pt"
        path_to_allele_vae_statedict = "../data/models/Indp/a_vae.pt"
        path_to_gp_statedict=None
    elif obj=='BA':
        path_to_peptide_vae_statedict = "../data/models/BA/p_vae.pt"
        path_to_allele_vae_statedict = "../data/models/BA/a_vae.pt"
        path_to_gp_statedict="../data/models/BA/gp.pt"
    elif obj=='PS':
        path_to_peptide_vae_statedict = "../data/models/PS/p_vae.pt"
        path_to_allele_vae_statedict = "../data/models/PS/a_vae.pt"
        path_to_gp_statedict="../data/models/PS/gp.pt"
assert not (not use_pretrain and guided), "Guided initialization cannot be performed without pretraining."
pep_obj = peptide_optimization.PeptideOptimization(
                                                path_to_peptide_vae_statedict =path_to_peptide_vae_statedict,
                                                path_to_allele_vae_statedict = path_to_allele_vae_statedict,
                                                path_to_gp_statedict = path_to_gp_statedict,
                                                obj=obj,
                                                task_id=task_id,
                                                track_with_wandb=TRACK_WITH_WANDB,
                                                wandb_entity="ENTITY",
                                                alpha=ALPHA,
                                                beta=BETA,
                                                gamma=GAMMA,
                                                delta=DELTA,
                                                zeta=ZETA,
                                                tempreture=0.2 if high else 0.5,
                                                num_initialization_points=100 if high else 20,
                                                max_n_oracle_calls=900 if high else 180,
                                                e2e_freq=10 if high else 0,
                                                k=50 if high else 10,
                                                allele=allele,
                                                seed=seed,
                                                use_pretrain=use_pretrain,
                                                init_n_update_epochs=init_epoch,
                                                num_update_epochs=2 if high else 5,
                                                bsz=bsz,
                                                guided=guided,
                                                force=force,
                                                vanilla=vanilla,
                                                high=high
                                                )

output_df = pep_obj.run_invbo()
if args.output is not None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    safe_allele= allele.replace("*", "_").replace(":", "-")
    out_put_dir=f'{args.output}/{safe_allele}_{obj}_{high}_{seed}_{vanilla}_{use_pretrain}_{guided}_{task_id}.csv'
    os.makedirs(args.output, exist_ok=True)
    output_df.to_csv(out_put_dir,index=False)