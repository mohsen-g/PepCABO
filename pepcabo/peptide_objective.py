import numpy as np
import torch 
from pepcabo.utils.pep_utils.seq_vae.model_positional_unbounded import InfoTransformerVAE
from pepcabo.utils.pep_utils.seq_vae.model_allele_ae import InfoCNNVAE2
from pepcabo.utils.pep_utils.seq_vae.data_loader import collate_fn, PeptideDataset, AlleleDataset
from pepcabo.latent_space_objective import LatentSpaceObjective
import pandas as pd
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


class PeptideObjective(LatentSpaceObjective):

    def __init__(
        self,
        obj,
        task_id,
        path_to_peptide_vae_statedict,
        path_to_allele_vae_statedict,
        xs_to_scores_dict={},
        max_string_length=1024,
        num_calls=0,
        predictor=None,
        allele_pscode = None,
        allele_name = None,
        use_pretrain = False
    ):
        self.obj = obj
        self.dim                    = 64 # VAE DEFAULT LATENT SPACE DIM
        self.path_to_peptide_vae_statedict  = path_to_peptide_vae_statedict # path to trained vae stat dict
        self.path_to_allele_vae_statedict = path_to_allele_vae_statedict
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.predictor = predictor
        self.allele_pscode = allele_pscode
        self.allele_name = allele_name
        self.use_pretrain = use_pretrain
        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
        )
        self.allele_pscode = self.a_vae.dataset.name_2_seq(self.allele_name)

    def vae_decode(self, z, max=False):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        if torch.cuda.is_available():
            z = z.cuda()
        peptide = z[:,:z.shape[1]//2]
        self.p_vae = self.p_vae.eval()

        if torch.cuda.is_available():
            self.p_vae = self.p_vae.cuda()
        peptide =peptide.reshape(-1, 1, self.p_vae.d_model)
        sample = self.p_vae.sample(z=peptide, max=max)
        decoded_peptide = [self.p_vae.dataset.decode(sample[i]) for i in range(sample.size(-2))]
        return decoded_peptide

    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        alleles = [self.allele_name]

        with suppress_stdout():
            results1 = self.predictor.predict(x, alleles,verbose=0)
        results1["affinity"] = 1 - np.log10(results1["affinity"])/(np.log10(50000))
        if self.obj=='BA':
            score = results1["affinity"].values
        elif self.obj=="PS":
            score = results1["presentation_score"].values
        else:
            raise
        fo = torch.tensor(results1[['affinity','processing_score','presentation_score']].values).float()

        score = torch.from_numpy(score).float() 
        score_dict = dict()
        for i in range(len(x)):
            score_dict[x[i]]=[score[i],fo[i].unsqueeze(0)]
        return score_dict

    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            used to tokenize inputs, etc. '''
        dataobj = PeptideDataset(pd.read_csv("../data/train_pairs.csv"))
        self.p_vae = InfoTransformerVAE(dataset=dataobj)
        if self.path_to_peptide_vae_statedict:
            state_dict = torch.load(self.path_to_peptide_vae_statedict)
            self.p_vae.load_state_dict(state_dict, strict=True) 
        if torch.cuda.is_available():
            self.p_vae = self.p_vae.cuda()
        self.p_vae = self.p_vae.eval()
        self.p_vae.max_string_length = self.max_string_length
        self.p_vae.kl_factor=0.1
        dataobj = AlleleDataset(pd.read_csv("../data/train_allele.csv"))
        self.a_vae = InfoCNNVAE2(dataset=dataobj)

        if self.path_to_peptide_vae_statedict:
            state_dict = torch.load(self.path_to_allele_vae_statedict)
            self.a_vae.load_state_dict(state_dict, strict=True) 

        for name, param in self.a_vae.named_parameters():
            if not (('fc' in name)  ):
                param.requires_grad = False
            else:
                param.requires_grad = False

        if torch.cuda.is_available():
            self.a_vae = self.a_vae.cuda()
        self.a_vae = self.a_vae.eval()

    def vaes_forward(self, xs_batch, allele_batch,use_pretrain,zs=None):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        if allele_batch is not None:
            P_list = []
            A_list = []
            A_embed = []
            for peptide,allele in zip(xs_batch,allele_batch):
                encoded_peptide = self.p_vae.dataset.encode(peptide).cuda()
                encoded_allele = self.a_vae.dataset.encode(allele).cuda()
                allels_embed = self.a_vae.dataset.embed(allele).cuda()

                    
                P_list.append(encoded_peptide)
                A_list.append(encoded_allele)
                A_embed.append(allels_embed)
            p_input = collate_fn(P_list)
            a_input = torch.stack(A_list)  
            a_embed = torch.stack(A_embed)
        else:
            P_list = []
            for peptide in xs_batch:
                if torch.cuda.is_available():
                    encoded_peptide = self.p_vae.dataset.encode(peptide).cuda()
                else:
                    encoded_peptide = self.p_vae.dataset.encode(peptide)
                P_list.append(encoded_peptide)
            p_input = collate_fn(P_list)



        p_dict = self.p_vae(p_input,zs=zs)
        _, z_p = p_dict['loss'], p_dict['z']
        z_p = z_p.reshape(-1,self.p_vae.d_model)

        if allele_batch is not None:
            a_dict = self.a_vae(a_input[:1],a_embed[:1])
            _, z_a = a_dict['loss'], a_dict['z']
            z_a = z_a.reshape(-1,self.a_vae.d_model).repeat(len(z_p),1)

            z = torch.cat((z_p,z_a),dim=1)
            loss_all = p_dict['recon_loss_all']
            kldiv_all = p_dict['kldiv']
        else:
            loss_all = p_dict['recon_loss_all']
            kldiv_all = p_dict['kldiv']
            z=z_p
        return z, None, loss_all, kldiv_all