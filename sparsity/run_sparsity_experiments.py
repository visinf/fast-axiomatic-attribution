# obtained from https://github.com/suinleelab/attributionpriors
# USAGE: python3 run_sparsity_experiments.py
import os, re, copy, pickle, sys
from collections import defaultdict

from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

import numpy as np
import pandas as pd
from scipy import linalg, special

from sklearn import (impute, preprocessing, model_selection, 
                     metrics, linear_model, datasets, pipeline)
import xgboost as xgb
import torch

from utils import ExpectedGradientsModel


# Global constants
REPS = 200
REFS = 1
EPOCHS = 100
NRUNS= 1
OUTPATH = 'results'
SUFFIX = '_all'

# Parameters for this run
RUN = 0

# Limit to a specific GPU
GPUS = [0]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=str(GPUS[(RUN%len(GPUS))])

# Inputs 
X = pd.read_csv('./X_sparsity.csv',index_col=0)
y = np.load('./y_sparsity.npy')

################
# FUNCTIONS
################

# Get random data splits
def get_data(rseed):
    Xtv, Xtest, ytv, ytest = model_selection.train_test_split(X,y,train_size=200,random_state=rseed)
    Xtrain, Xvalid, ytrain, yvalid = model_selection.train_test_split(Xtv,ytv,train_size=100,random_state=rseed)
    
    imp = impute.SimpleImputer()
    ss = preprocessing.StandardScaler()

    Xtrain_imp = imp.fit_transform(Xtrain)
    Xvalid_imp = imp.transform(Xvalid)
    Xtest_imp = imp.transform(Xtest)
    
    Xtrain_ss = ss.fit_transform(Xtrain_imp)
    Xvalid_ss = ss.transform(Xvalid_imp)
    Xtest_ss = ss.transform(Xtest_imp)
    
    return Xtrain_ss, Xvalid_ss, Xtest_ss, ytrain, yvalid, ytest

def tensorize(arr):
    ret = torch.from_numpy(arr).float().cuda()
    if len(arr.shape)==1:
        ret = ret.reshape(-1,1)
    return ret

# Gini coefficient functions
# Torch
def tgini(x):
    mad = torch.mean(torch.abs(x.reshape(-1,1)-x.reshape(1,-1)))
    rmad = mad/torch.mean(x)
    g = 0.5*rmad
    return g

# Numpy
def ngini(x):
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

def lorenz_curve(x):
    curve = np.cumsum(np.sort(x))
    return curve/curve[-1]

# Torch network code

def get_layers(hid_layers,dropout=0.0,bias=True):
    layerlist = [torch.nn.Linear(Xtrain_tensor.shape[1],hid_layers[0], bias=bias)]
    if len(hid_layers)!=1:
        for i in range(len(hid_layers)-1):
            if dropout>0:
                layerlist.append(torch.nn.Dropout(p=dropout))
            layerlist.extend([torch.nn.ReLU(),torch.nn.Linear(hid_layers[i],hid_layers[i+1], bias=bias)])
    if dropout>0:
        layerlist.append(torch.nn.Dropout(p=dropout))
    layerlist+=[torch.nn.ReLU(),torch.nn.Linear(hid_layers[-1],1, bias=bias)]
    return torch.nn.Sequential(*layerlist)

def get_model_opt(hid_layers,dropout=0.0, bias=True):
    base_model = get_layers(hid_layers,dropout=dropout, bias=bias)
    augmented_model = ExpectedGradientsModel(base_model.cuda(),refset)
    optimizer = torch.optim.Adam(augmented_model.parameters(),lr=learning_rate)
    return augmented_model, optimizer

# Gets gradients of raw model output (logits)
def get_grads(m,x):
    c = x.clone()
    c.requires_grad = True
    yp = m(c)
    return torch.autograd.grad(outputs=yp,inputs=c,grad_outputs=torch.ones_like(yp).cuda(),create_graph=True)[0]

def get_input_x_grads(m,x):
    c = x.clone()
    c.requires_grad = True
    yp = m(c)
    return torch.autograd.grad(outputs=yp,inputs=c,grad_outputs=torch.ones_like(yp).cuda(),create_graph=True)[0] * x

# Gets gradients of summed model log-probability outputs
def get_grads_ross(m,x):
    c = x.clone()
    c.requires_grad = True
    yp = m(c)
    logits = yp
    logprobs = torch.nn.functional.logsigmoid(logits)
    sumlogprobs = 2*logprobs-logits  # Gives sum of lob probabilities from positive class log-probs and logits
    return torch.autograd.grad(outputs=sumlogprobs,inputs=c,grad_outputs=(torch.ones_like(yp)).cuda(),create_graph=True)[0]

# Regularizers/priors
def l1_params_all(model,shaps,grads):
    all_params = torch.cat([x.view(-1) for x in model.parameters() if len(x.shape)!=1])
    return torch.mean(torch.abs(all_params))

def sgl_penalty(arr):
    l1_indiv = arr.abs().sum()
    l1_group = torch.norm(arr,dim=0).sum()*np.sqrt(float(arr.shape[0]))
    return l1_indiv+l1_group
def sgl_params_all(model,shaps,grads):
    all_weights = [x for x in model.parameters() if len(x.shape)!=1]
    all_biases = [x for x in model.parameters() if len(x.shape)==1]
    weight_penalties = torch.stack([sgl_penalty(w) for w in all_weights]).sum()
    bias_penalties = torch.stack([2*b.abs().mean() for b in all_biases]).sum()
    return weight_penalties+bias_penalties
def sgl_params_first(model,shaps,grads):
    params = next(iter(model.parameters()))
    return sgl_penalty(params)
    
def l1_grad(model,shaps,grads):
    return grads.abs().mean()

def gini_grad(model,shaps,grads):
    abs_attrib = grads.abs()
    return -tgini(abs_attrib.mean(0))#+abs_attrib.mean()

def l1_ross(model,shaps,grads):
    return grads.abs().mean()

def gini_ross(model,shaps,grads):
    abs_attrib = grads.abs()
    return -tgini(abs_attrib.mean(0))#+abs_attrib.mean()

def gini_input_x_grad(model,shaps,grads):

    abs_attrib = grads.abs()
    return -tgini(abs_attrib.mean(0))#+abs_attrib.mean()

def l1_input_x_grad(model,shaps,grads):
    return grads.abs().mean()

def gini_eg(model,shaps,grads):
    abs_attrib = shaps.abs()
    return -tgini(abs_attrib.mean(0))

def gini_mixed(model,shaps,grads):
    abs_attrib = shaps.abs()
    return -tgini(abs_attrib.mean(0))+abs_attrib.mean()

def l2_params_all(model,shaps,grads):
    all_params = torch.cat([x.view(-1) for x in model.parameters() if len(x.shape)!=1])
    return torch.norm(all_params)

def l2_params_first(model,shaps,grads):
    params = next(iter(model.parameters())).view(-1)
    return torch.norm(params)

def l1_params_first(model,shaps,grads):
    params = next(iter(model.parameters())).view(-1)
    return torch.mean(torch.abs(params))

def l1_eg(model,shaps,grads):
    return shaps.abs().mean()

def l2_eg(model,shaps,grads):
    return torch.norm(shaps)

# Train model
def train(nrounds,model,optimizer,aux_losses=[],alpha=1.0,require_grads=False,require_attributions=False,iterator=tqdm,k=1):
    # Cache model state
    init_state = model.training
    
    # Set up
    results = []
    best_model = None
    best_val = float('-inf')
    itr = iterator(range(nrounds))
    for t in itr:
        # Train mode
        model.train()
        
        # Get train predictions
        if require_attributions:
            train_pred, train_shap = model(Xtrain_tensor,shap_values=True,k=k)
        else:
            train_pred = model(Xtrain_tensor)
            train_shap = None
        if 'input_x_grad' in str(aux_losses):
            train_grads = get_input_x_grads(model,Xtrain_tensor) if require_grads else None
        elif 'ross' in str(aux_losses):
            train_grads = get_grads_ross(model,Xtrain_tensor) if require_grads else None
        else:
            train_grads = get_grads(model,Xtrain_tensor) if require_grads else None
        # Base losses
        loss = loss_fn(train_pred,ytrain_tensor)
        total_loss = loss
        
        # Extra losses
        aux_loss_results = [aux_loss(model,train_shap,train_grads) for aux_loss in aux_losses]
        aux_loss_values = [r.item() for r in aux_loss_results]
        for r in aux_loss_results:
            total_loss += alpha*r
        
        # Eval mode
        model.eval()
            
        # Validation predictions
        valid_pred = model(Xvalid_tensor).detach().cpu().numpy()
        val_loss = metrics.roc_auc_score(yvalid_tensor.detach().cpu().numpy(),valid_pred)
        
        # Cache results
        results.append(tuple([loss.item(),val_loss]+aux_loss_values))
        if val_loss>best_val:
            best_model = copy.deepcopy(model)
            best_val = val_loss
            
        # Update progress bar
        aux_string = "|".join([f'{aux:.2f}' for aux in aux_loss_values])
        if iterator is tqdm:
            desc = f'Trn:{loss.item():.2f}/Val:{val_loss:.2f}/Aux:{aux_string}'
            itr.set_description(desc)
        
        # Update model
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    if init_state: best_model.train()
    else: best_model.eval()
    return (best_model,best_val,np.array(results))

################################################
# Old code: find best unreg. architecture
################################################

# Fix best unreg. architecture from full data
best_layers=[512,128,32]

################################################
# Train models with all penalties
################################################

# Loss and LR
loss_fn = torch.nn.BCEWithLogitsLoss()
learning_rate=1e-3


# Train unregularized models
unreg_models = []
for i in tqdm(range(REPS),desc=f'unregularized'):
    (Xtrain_tensor, Xvalid_tensor, Xtest_tensor, 
     ytrain_tensor, yvalid_tensor, ytest_tensor) = [tensorize(arr) for arr in get_data(i)]
    ytest = ytest_tensor.detach().cpu().numpy()
    refset = torch.utils.data.TensorDataset(torch.cat([Xtrain_tensor]*REFS))
    model, opt = get_model_opt(best_layers, bias=True)
    model,val_score,per_epoch = train(EPOCHS,model,opt,aux_losses=[],iterator=iter,require_attributions=False,require_grads=False)
    model.eval()
    baseline_train = torch.zeros(Xtrain_tensor.shape).cuda()
    baseline_valid = torch.zeros(Xvalid_tensor.shape).cuda()
    test_score = metrics.roc_auc_score(ytest,model(Xtest_tensor).detach().cpu().numpy())
    unreg_models.append((val_score,test_score,per_epoch))
with open(f'{OUTPATH}/{REPS}reps_{RUN}_unreg.pkl','wb') as w:
    pickle.dump(unreg_models,w)

# Train unregularized models w/o a bias
unreg_models = []
for i in tqdm(range(REPS),desc=f'unregularized'):
    (Xtrain_tensor, Xvalid_tensor, Xtest_tensor, 
     ytrain_tensor, yvalid_tensor, ytest_tensor) = [tensorize(arr) for arr in get_data(i)]
    ytest = ytest_tensor.detach().cpu().numpy()
    refset = torch.utils.data.TensorDataset(torch.cat([Xtrain_tensor]*REFS))
    model, opt = get_model_opt(best_layers, bias=False)
    model,val_score,per_epoch = train(EPOCHS,model,opt,aux_losses=[],iterator=iter,require_attributions=False,require_grads=False)
    model.eval()
    baseline_train = torch.zeros(Xtrain_tensor.shape).cuda()
    baseline_valid = torch.zeros(Xvalid_tensor.shape).cuda()
    test_score = metrics.roc_auc_score(ytest,model(Xtest_tensor).detach().cpu().numpy())
    unreg_models.append((val_score,test_score,per_epoch))
with open(f'{OUTPATH}/{REPS}reps_{RUN}_unreg_nobias.pkl','wb') as w:
    pickle.dump(unreg_models,w)
      
# Train other models
reg_strengths = [1e-2, 1e-1, 1e+0, 1e+1, 1e+2]
models = defaultdict(list)
model_args = {
            'giniross':{'aux_losses':[gini_ross],'require_attributions':False,'require_grads':True},
            'ginigrad':{'aux_losses':[gini_grad],'require_attributions':False,'require_grads':True},
            'giniinputxgrad_nobias':{'aux_losses':[gini_input_x_grad],'require_attributions':False,'require_grads':True},
            'ginieg':{'aux_losses':[gini_eg],'require_attributions':True,'require_grads':False},
}
#for alpha in reg_strengths[chunk_start:chunk_end]:
for alpha in reg_strengths:
    rep_results = defaultdict(list)
    for i in tqdm(range(REPS),desc=f'alpha={alpha:.3e}'):
        (Xtrain_tensor, Xvalid_tensor, Xtest_tensor, 
         ytrain_tensor, yvalid_tensor, ytest_tensor) = [tensorize(arr) for arr in get_data(i)]
        ytest = ytest_tensor.detach().cpu().numpy()
        refset = torch.utils.data.TensorDataset(torch.cat([Xtrain_tensor]*REFS))
        for k in model_args:
            if 'nobias' in k:
                init_model, opt = get_model_opt(best_layers, bias=False)
            else:
                init_model, opt = get_model_opt(best_layers, bias=True)
            model, val_score, per_epoch = train(EPOCHS,init_model,opt,alpha=alpha,iterator=iter,k=REFS,**model_args[k])
            model.eval()
            baseline_train = torch.zeros(Xtrain_tensor.shape).cuda()
            baseline_valid = torch.zeros(Xvalid_tensor.shape).cuda()
            test_score = metrics.roc_auc_score(ytest,model(Xtest_tensor).detach().cpu().numpy())
            rep_results[k].append((val_score,test_score,per_epoch))
    for k in rep_results: models[k].append(rep_results[k])
with open(f'{OUTPATH}/{REPS}reps_{RUN}{SUFFIX}.pkl','wb') as w:
    pickle.dump(models,w)


egmodels = defaultdict(list)
#run eg with more ref samples
reg_strengths = [1e+0] #best reg strength from running with REFS = 1
refs = [1,2,4,8,16,32,64,128]
for alpha in reg_strengths:
    rep_results = defaultdict(list)
    for i in tqdm(range(REPS),desc=f'alpha={alpha:.3e}'):
        (Xtrain_tensor, Xvalid_tensor, Xtest_tensor, 
        ytrain_tensor, yvalid_tensor, ytest_tensor) = [tensorize(arr) for arr in get_data(i)]
        ytest = ytest_tensor.detach().cpu().numpy()
        for ref in refs:
            refset = torch.utils.data.TensorDataset(torch.cat([Xtrain_tensor]*ref))
            k = 'ginieg' + str(int(ref))
            
            init_model, opt = get_model_opt(best_layers, bias=True)
            
            model, val_score, per_epoch = train(EPOCHS,init_model,opt,alpha=alpha,iterator=iter,k=ref,**{'aux_losses':[gini_eg],'require_attributions':True,'require_grads':False})
            model.eval()
            baseline_train = torch.zeros(Xtrain_tensor.shape).cuda()
            baseline_valid = torch.zeros(Xvalid_tensor.shape).cuda()
            test_score = metrics.roc_auc_score(ytest,model(Xtest_tensor).detach().cpu().numpy())
            rep_results[k].append((val_score,test_score,per_epoch))
    for k in rep_results: egmodels[k].append(rep_results[k])
with open(f'{OUTPATH}/{REPS}reps_{RUN}_eg_refs_powers_of_two.pkl','wb') as w:
    pickle.dump(egmodels,w)