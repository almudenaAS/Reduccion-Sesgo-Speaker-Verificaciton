import argparse
from dca_plda import calibration
from dca_plda import scores as dscores
import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import re
from dca_plda.utils import load_configs, get_class_to_cluster_map_from_config
from dca_plda.utils_for_scripts import setup_torch_and_cuda_vars, print_graph, train, mkdirp
from dca_plda.data import LabelledDataset
from dca_plda.modules import DCA_PLDA_Backend, Hierarchical_DCA_PLDA_Backend
from dca_plda.utils_for_scripts import np_to_torch, evaluate, load_model
from dca_plda.scores import IdMap, compute_performance


def print_perf(pos, neg, name, ptars):

    det           = dscores.Det(pos, neg, pav=True)
    min_dcfs      = det.min_dcf(ptars)
    act_dcfs      = det.act_dcf(ptars)
    act_cllrs     = det.act_cllr(ptars)
    min_cllrs     = det.min_cllr(ptars)
    min_cllrs_pav = det.min_cllr(ptars, with_pav=True)
    eer           = det.eer(from_rocch=True)

    print("%-22s |  %6d  %6d |  %6.2f  |  %7.4f  %7.4f  %7.4f  |  %7.4f  %7.4f  %7.4f  | %7.4f %7.4f | %7.4f %7.4f "%
          (name,
           len(det.tar),
           len(det.non),
           eer*100,
           act_cllrs[0], min_cllrs[0], min_cllrs_pav[0],
           act_cllrs[1], min_cllrs[1], min_cllrs_pav[1],
           act_dcfs[0], min_dcfs[0],
           act_dcfs[1], min_dcfs[1]))

def train_score_calibration(key, ptar, fmt, scores, model):
    print("Train a post-hoc score calibrator using linear logistic regression.")
# '--key',      help='Key used for calibration. Required when scores are in h5 format.', default=None)
# '--ptar',     help='Effective prior for positive samples used during training. This parameter should coincide roughly (ie, in order of magnitude) to the effective prior that will be used to evaluate the system but it usually does not need to be identical.', default=0.5, type=float)
# '--fmt',      help='Format of input scores. If set to "npz", an npz file with two arrays, pos and neg, for the positive and negative samples is expected.', default='npz')
# 'scores',     help='Score file in h5 or npz format.')
# 'model',      help='File where to print the calibration parameters, alpha (scale) and beta (shift).')

    if fmt == "h5":
        assert key != None
        scores = dscores.Scores.load(scores)
        key = dscores.Key.load(key)
        ascores = scores.align(key)
        pos = ascores.score_mat[key.mask==1]
        neg = ascores.score_mat[key.mask==-1]
                                                                        
    else:
        sc = np.load(scores)
        pos, neg = sc['pos'], sc['neg'] 

    a, b = calibration.logregCal(pos, neg, return_params=True, Ptar=ptar)
    np.savez(model, a=a, b=b)

    print("Trained linear calibration model with %d positive and %d negative samples."%(len(pos), len(neg)))
    print("Alpha = %.4f"%a)
    print("Beta  = %.4f"%b)

def eval_score_calibration(key, ptar, fmt, scores, model, outscores):
    print("Train a post-hoc score calibrator using linear logistic regression.")
# '--key',      help='Key used for evaluation. If provided when scores are in h5 format, performance metrics before and after calibration will be printed..', default=None)
# '--ptar',     help='Effective prior for positive samples used for computing evaluation metrics.', default=0.01, type=float)
# '--fmt',      help='Format of input scores. If set to "npz", an npz file with two arrays, pos and neg, for the positive and negative samples is expected, or a single array with all scores. In the latter case, performance metrics are not computed', default='npz')
# 'scores',     help='Score file in h5 or npz format.')
# 'model',      help='File with the calibration parameters, alpha (scale) and beta (shift).')
# 'outscores',  help='File name for the output scores, without extension. Will be saved in the same format as the input with the correct extension.')

    if model != "":
        model = np.load(model)
        a, b = model['a'], model['b']
    else:
        a,b = 0.5, 0.5

    # Read and calibrat scores
    has_key = False
    if fmt == "h5":
        if key != None:
            key = dscores.Key.load(key)
            has_key = True
            
        raw_scores = dscores.Scores.load(scores)
        if has_key:
            ascores = raw_scores.align(key)
            raw_pos = ascores.score_mat[key.mask==1]
            raw_neg = ascores.score_mat[key.mask==-1]        

        cal_scores = raw_scores
        cal_scores.score_mat = a * raw_scores.score_mat + b
        if has_key:
            ascores = cal_scores.align(key)
            cal_pos = ascores.score_mat[key.mask==1]
            cal_neg = ascores.score_mat[key.mask==-1]        

        cal_scores.save("%s.h5"%outscores)
           
    else:
        sc = np.load(scores)
        if 'all' in sc.files:
            raw_all = sc['all']
            cal_all = a * raw_all + b
            np.savez("%s.npz"%outscores, all=cal_all)

        else:
            raw_pos, raw_neg = sc['pos'], sc['neg'] 
            cal_pos = a * raw_pos + b
            cal_neg = a * raw_neg + b
            has_key = True
            np.savez("%s.npz"%outscores, pos=cal_pos, neg=cal_neg)

            
    # Compute performance before and after calibration
    if has_key:
        print("Results before and after calibration on test %d positive and %d negative samples"%(len(raw_pos), len(raw_neg)))
        ptars = [ptar, 0.5]
        print("%-22s                              |         Ptar=%4.2f           |         Ptar=%4.2f           |    Ptar=%4.2f    |   Ptar=%4.2f  "%(" ",ptars[0], ptars[1], ptars[0], ptars[1]))
        print("%-22s |   #TGT   #IMP   |    EER   |   ACLLR MCLLR_LIN MCLLR_PAV |   ACLLR MCLLR_LIN MCLLR_PAV |  ADCF    MDCF   |   ADCF   MDCF  "%"Key")   
        print_perf(raw_pos, raw_neg, "before calibration", ptars)
        print_perf(cal_pos, cal_neg, "after calibration",  ptars)


def train_calibration(debug, cuda, seed, configs, mods, init_subset, restart, print_min_loss, trn_embeddings, trn_metafile, dev_table, out_dir):
# '--debug',        help='Enable debug mode.', action='store_true')
# '--cuda',         help='Enable cuda.', action='store_true')
# '--seed',         help='Seed used for training.', default=0, type=int)
# '--configs',      help='List of configuration files to load. They are loaded in order, from left to right, overwriting previous values for repeated parameters.', default=None)
# '--mods',         help='List of values to overwride config parameters. Example: training.num_epochs=20,architecture.lda_dim=200', default=None)
# '--init_subset',  help='Subset of the train files to be used for initialization. For default, the files in trn_metafile are used.', default=None)
# '--restart',      help='Restart training from last available model.', action='store_true')
# '--print_min_loss', help='Print the min loss for each dev set at each epoch.', action='store_true')
# 'trn_embeddings', help='Path to the npz file with training embeddings.')
# 'trn_metafile',   help='Path to the metadata for the training samples (all samples listed in this file should be present in the embeddings file).')
# 'dev_table',      help='Path to a table with one dev set per line, including: name, npz file with embeddings, key file, and durations file (can be missing if not using duration-dependent calibration).')
# 'out_dir',        help='Output directory for models.')
    
    default_config = {
        'architecture':{
            'lda_dim': 200},
        'training': {
            'loss': 'cross_entropy',
            'ptar': 0.01,
            'max_norm': 4,
            'l2_reg': 0.0001,
            'learning_rate': 0.0005,
            'betas': (0.5, 0.99),
            'learning_rate_params': None,
            'init_params': {'w_init': 0.5},
            'batch_size': 256,
            'num_epochs': 50,
            'num_samples_per_class': 2,
            'num_batches_per_epoch': 1000,
            'compute_ave_model': False}}

    mkdirp(out_dir)
    ##### Read the configs
    config = load_configs(configs, default_config, mods, "%s/config"%(out_dir))
    ##### Set the seed 
    seed = random.randint(1, 10000) if seed is None else seed
    print("Using seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    ##### Set the device and default data type
    device = setup_torch_and_cuda_vars(cuda)

    ###### Load the dataset and create the model object
    cluster_ids = get_class_to_cluster_map_from_config(config.architecture)
    trn_dataset = LabelledDataset(trn_embeddings, trn_metafile, cluster_ids=cluster_ids)
    in_size = trn_dataset[0]['emb'].shape[0]

    if cluster_ids:
        model = Hierarchical_DCA_PLDA_Backend(in_size, config.architecture).to(device)
    else:   
        model = DCA_PLDA_Backend(in_size, config.architecture).to(device)

    print_graph(model, trn_dataset, device, out_dir)

    ###### Train
    print("\n####################################################################################")
    print("Starting training")

    train(model, trn_dataset, config.training, dev_table, out_dir, 
        device, seed, restart, debug, init_subset, print_min_loss)


def eval_calibration(cuda, durs, min_dur, keylist, ptar, set, raw, level, cluster_priors, fmt, model, embeddings, enroll_map, test_map, out_dir):
# ('--cuda',     help='Enables cuda.', action='store_true')
# ('--durs',     help='File with durations, needed only if the model was created with duration_dependent_calibration = True.', default=None)
# ('--min_dur',  help="Minimum duration in seconds below which all scores for a model or test are turned to 0.", default=0, type=float)
# ('--keylist',  help='List of keys for scoring. If not provided, only the scores are printed in the out_dir.', default=None)
# ('--ptar',     help='Prior for Cllr and DCF computation if keylist is provided.', default=0.01, type=float)
# ('--set',      help='Name for the set, to be used in the results file.', default=None)
# ('--raw',      help='Output the raw scores, before the calibration stages (used for analysis).', default=False, action='store_true')
# ('--level',    help='Which LLRs to return for hierarchical DCA-PLDA: None (LLRs for the classes), level1 (LLRs for clusters), level2 (LLRs for clusters conditioned on known cluster).', default=None)
# ('--cluster_priors', help='Only for hierarchical DCA-PLDA. A table with the detection prior for each cluster. If provided, the output LLRs will be computed using these priors instead of the default ones.', action=None)
# ('--fmt',      help='Format of output scores: h5 or ascii.', default='h5')
# ('model',      help='Path to the model to be used for evaluation.')
# ('embeddings', help='Path to the npz file with development embeddings.')
# ('enroll_map', help='Map from enrollment ids (first column) to the ids used in the embeddings file (second column).\
#                                         The mapping could be one to many for multi-session enrollment.')
# ('test_map',   help='Map from test ids (first column) to the ids used in the embeddings file (second column).')
# ('out_dir',    help='Output directory for scores and results, if keylist is provided.')

    mkdirp(out_dir)

    ##### Set the device and data type
    device = setup_torch_and_cuda_vars(cuda)

    ###### Load the model
    model = load_model(model, device)
    print("Loaded model from %s"%model)

    ###### Load the data
    dataset = LabelledDataset(embeddings, durs, meta_is_dur_only=True, device=device, skip_missing=True)
    if model.enrollment_classes is not None:
        assert enroll_map == 'NONE'
        if level == 'level1':
            emap = IdMap.load('NONE', model.level1_detector.enrollment_classes)
        else:
            emap = IdMap.load('NONE', model.enrollment_classes)
        eids = emap.model_ids
    else:
        emap = IdMap.load(enroll_map, dataset.get_ids())
        eids = None

    tmap = IdMap.load(test_map, dataset.get_ids())
        
    ###### Generate the scores
    cluster_prior_dict = dict(np.loadtxt(cluster_priors, dtype='O', converters={1: float})) if cluster_priors else None
    scores = evaluate(model, dataset, emap, tmap, min_dur=min_dur, raw=raw, level=level, cluster_prior_dict=cluster_prior_dict)
    scores.save("%s/scores.%s"%(out_dir, fmt), fmt=fmt)

    if keylist is not None:
        compute_performance(scores, keylist, "%s/results"%out_dir, ptar=ptar, setname=set, enrollment_ids=eids)
        