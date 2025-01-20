import calvados as cal
import numpy as np
import pandas as pd
from scripts.predictor import *
from scripts.mc_seq import *
import joblib
import pickle
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--start',nargs='?',required=True,type=int)
parser.add_argument('--pswap',nargs='?',required=True,type=float)
parser.add_argument('--constr',nargs='*',required=False,type=str)
args = parser.parse_args()

residues = pd.read_csv('scripts/residues_CALVADOS2.csv').set_index('one')
nu_file = 'scripts/svr_model_nu.joblib'

features = ['mean_lambda', 'faro', 'shd', 'ncpr', 'fcr', 'scd', 'ah_ij','nu_svr']
features_clean = {
    'mean_lambda' : 'lambda',
    'faro' : 'f(aromatics)',
    'shd' : 'SHD',
    'ncpr' : 'NCPR',
    'fcr' : 'FCR',
    'scd' : 'SCD',
    'ah_ij' : 'LJ pairs',
    'nu_svr' : 'nu(SVR)'
}

print('Input features are:')
print('>>>>> '+ ', '.join([features_clean[fe] for fe in features]))

models = {}
models['dG'] = joblib.load(f'scripts/model_dG.joblib')
# models['logcdil_mgml'] = joblib.load(f'scripts/model_logcdil_mgml.joblib')
model = models['dG']

mltype = 'mlp'
alpha = 5
layers = (10,10)

print('Models loaded.')

CHARGE_TERMINI = True

####

df_full = pd.read_csv('IDRome_DB_full.csv').set_index('id')
# df_full = add_dG_pred_to_df(df_full, 'dG', model, features)

# Load nu, dG_pred
df_nu = df_full.loc[df_full['dG_pred'] > -4]
df_nu = df_nu.loc[df_nu['dG_pred'] < -1]
df_nu = df_nu.loc[df_nu['nu_svr'] < 0.5] # 0.53

seq_names = df_nu['seq_name'].values
nus_all = df_nu['nu_svr'].values
dGs_all = df_nu['dG_pred'].values
seqs_all = df_nu['fasta'].values

####

ah_intgrl_map = cal.sequence.make_ah_intgrl_map(residues)
lambda_map = cal.sequence.make_lambda_map(residues)

print(args.constr)

constraints = {}

tolerances = {
        'nu_svr' : 0.001,
        'mean_lambda' : 0.002,
        'ah_ij' : 0.004,
        'ncpr' : 0.002,
        'scd' : 0.05
}

for feat in features:
    if args.constr == None:
        break
    if feat in args.constr:
        constraints[feat] = {
                'value' : None,
                'tolerance' : tolerances[feat],
                }

print(constraints)

target = {
    'feat' : 'dG',
    'value' : -10.,
    'k' : 0.3
}

str_constr = ''
for constr in constraints:
    str_constr += f'_{constr}'
if len(str_constr) == 0:
    str_constr = '_none'

#####

nseqs = 200
verbose = False
steps_per_iter = 5
niter = 200
pswap = args.pswap
psubst = 1. - pswap

seqs_subset = random.choices(seqs_all, k = nseqs)

dir_path = f'output/constr{str_constr}/pswap{pswap:.2f}'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

for idx, seq in enumerate(seqs_subset):
    jdx = idx+args.start
    for key in constraints.keys():
        constraints[key]['value'] = None
    res = mc_along_diag(seq, features, residues, nu_file,
                      target, constraints, model,
                      steps_per_iter = steps_per_iter,
                      niter = niter,
                      pswap = pswap, psubst = psubst,
                      a = 100., ah_intgrl_map = ah_intgrl_map,
                      lambda_map=lambda_map,
                        verbose=verbose)
    fout = f'{dir_path}/run_{jdx}.pkl'
    with open(fout, 'wb') as f:
        pickle.dump(res, f)
