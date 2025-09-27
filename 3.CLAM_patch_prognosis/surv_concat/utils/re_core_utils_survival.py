import os
import torch
import numpy as np
import pandas as pd
from utils.utils_survival import *
from datasets_CLAM.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from lifelines.utils import concordance_index 
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def coxph_loss(y_pred, y_time, y_event):
    time = y_time
    event = y_event

    sort_time = torch.argsort(time, 0, descending=True)
    event = torch.gather(event, 0, sort_time)
    
    risk = torch.gather(y_pred, 0, sort_time)
    exp_risk = torch.exp(risk)
    log_risk = torch.log(torch.cumsum(exp_risk, 0))
    censored_likelihood = (risk - log_risk) * event
    censored_likelihood = torch.sum(censored_likelihood)
    censored_likelihood = censored_likelihood / y_time.shape[0]
    return -censored_likelihood

def c_index(y_pred, y_time, y_event):
    time = y_time
    event = y_event

    return concordance_index(time, -np.exp(y_pred), event)

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, version, fold, args):
    """   
        train for a single fold
    """
    print('\nTraining Version {} & {}!'.format(version, fold))
    writer_dir = os.path.join(args.results_dir, f"{version}_{fold}")
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/test splits...', end=' ')
    train_split, test_split = datasets
    # save_splits(datasets, ['train', 'test'], os.path.join(args.results_dir, 'Version_{}_{}_split.csv'.format(version, fold)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        if args.n_classes == 2:
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0,10.0]).to('cuda'))
        else:
            loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"encoding_size": args.encoding_size, "dropout": args.drop_out}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss(weight=[1.0,6.0])
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    early_stopping = None
    
    model_save_path = os.path.join(args.results_dir, 'model_weights')
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_weight_path = os.path.join(model_save_path, f"s_{version}_{fold}_checkpoint.pt")
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path))  # 加载权重
        print(f"Loaded weights from {model_weight_path}")
    else:
        print(f"Weight file not found: {model_weight_path}; We need re-training!")
        for epoch in range(args.max_epochs):
            print("Epoch:", epoch)
            train_loop_clam(epoch, model, train_loader, optimizer)
    
        torch.save(model.state_dict(), os.path.join(model_save_path, f"s_{version}_{fold}_checkpoint.pt"))

    test_c_index, test_auc, test_KM_df = summary(model, train_loader, test_loader, args.n_classes)
    print('Version:{}, Fold:{}, Test C-Index: {:.4f}, 3-year AUC: {:.4f}'.format(version, fold, test_c_index, test_auc))

    return test_c_index, test_auc, test_KM_df


def train_loop_clam(epoch, model, loader, optimizer):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    print('\n')
    
    WSI_logits = []
    slide_time = []
    slide_event = []
    
    Cindex_logits = []
    Cindex_time = []
    Cindex_event = []

    step = 0
    gc = 0
    epochCindex = 0
    epochCoxloss = 0
    
    for batch_idx, batch in enumerate(loader):
        
        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            data, cluster_id, label = data.to(device), cluster_id, label.to(device)
        else:
            ###====
            data, label,y_time,y_event = batch
            data, label = data.to(device), label.to(device)
            cluster_id = None
        # if dropinput > 0:
        #     data = F.dropout(data, p=dropinput)
        # # print('================',label)
        logits, Y_prob, Y_hat, _, instance_dict = model(h=data, label=label, instance_eval=True)
        # logits, Y_prob, Y_hat, _, instance_dict = model(h=data, label=label, instance_eval=True)
###============
        WSI_logits.append(logits[0][0].clone())
        slide_time.append(y_time[0].item())
        slide_event.append(y_event[0].item())
        
        
        Cindex_logits.append(logits[0][0].item())
        Cindex_time.append(y_time[0].item())
        Cindex_event.append(y_event[0].item())
        step +=1
        if(step!=0 and step%10 ==0):
                
                WSI_logits = torch.stack(WSI_logits)
                # print('=======WSI_logits2222=====',WSI_logits)
                slide_time = torch.tensor(slide_time).to('cuda:0')
                slide_event = torch.tensor(slide_event).to('cuda:0')

                coxloss = coxph_loss(WSI_logits,slide_time,slide_event)
                # print('===coxloss======',coxloss)
                coxloss.backward()
                # step

                epochCoxloss += coxloss
                WSI_logits = []
                slide_time = []
                slide_event = []
                gc +=1
                if(gc!=0 and gc%20 == 0):
                    optimizer.step()
                    optimizer.zero_grad()
    epochCoxloss = epochCoxloss*25/len(loader)

    cindex = c_index(Cindex_logits,Cindex_time,Cindex_event)
    print('Train: Epoch: {}, coxloss: {:.4f} cindex: {:.4f}'.format(epoch, epochCoxloss, cindex))


def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        WSI_logits = []
        slide_time = []
        slide_event = []
        for batch_idx, batch in enumerate(loader):
            if hasattr(model, "num_clusters"):
                data, cluster_id, label = batch
                data, cluster_id, label = data.to(device), cluster_id, label.to(device)
            else:
                data, label,y_time,y_event = batch
                data, label = data.to(device), label.to(device)
                cluster_id = None
            logits, Y_prob, Y_hat, _, instance_dict = model(h=data, cluster_id=cluster_id, label=label, instance_eval=True)
            # logits, Y_prob, Y_hat, _, instance_dict = model(h=data,label=label, instance_eval=True)
            WSI_logits.append(logits[0][0])
            slide_time.append(y_time[0].item())
            slide_event.append(y_event[0].item())
            
    WSI_logits = torch.tensor(WSI_logits)
    slide_time = torch.tensor(slide_time)
    slide_event = torch.tensor(slide_event)

    coxloss = coxph_loss(WSI_logits,slide_time,slide_event)
    cindex = c_index(WSI_logits,slide_time,slide_event)
    print('Valid: Epoch: {}, coxloss: {:.4f}cindex: {:.4f}'.format(epoch,coxloss, cindex))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, coxloss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, train_loader, test_loader, n_classes):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    
    train_case_ids = train_loader.dataset.slide_data['TCGA_case_id']
    train_case_time = train_loader.dataset.slide_data['Days']
    train_case_event = train_loader.dataset.slide_data['Censor']
    
    train_surv_df = pd.DataFrame(
        data = np.column_stack((train_case_time, train_case_event)),
        index = train_case_ids,
        columns = ['Survival', 'Censor']
    )
    train_agg_surv_df = train_surv_df.groupby(train_surv_df.index).first()

    test_slide_ids = test_loader.dataset.slide_data['HE_entity_submitter_id']
    test_case_ids = test_loader.dataset.slide_data['TCGA_case_id']

    patient_results = {}
    WSI_logits = []
    slide_time = []
    slide_event = []

    for batch_idx, batch in enumerate(test_loader):
        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
        else:
            data, label, y_time, y_event  = batch
            data, label = data.to(device), label.to(device)
            cluster_id = None
        
        # slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)
            WSI_logits.append(logits[0][0].item())
            slide_time.append(y_time[0].item())
            slide_event.append(y_event[0].item())
    
    target_surv_df = pd.DataFrame(
        data = np.column_stack((slide_time, slide_event)),
        index = test_case_ids,
        columns = ['Survival', 'Censor']
    )
    
    predict_surv_df = pd.DataFrame(
        data = np.array(WSI_logits),
        index = test_case_ids,
        columns = ['Predicted_risk']
    )
    
    pred_agg_risks_df = predict_surv_df.groupby(predict_surv_df.index).max()
    target_agg_surv_df = target_surv_df.groupby(target_surv_df.index).first()
    
    train_X_structured = Surv.from_dataframe('Censor', 'Survival', train_agg_surv_df)
    test_X_structured = Surv.from_dataframe('Censor', 'Survival', target_agg_surv_df)
    
    times = [365, 730, 1095]
    test_aucs, test_mean_auc = cumulative_dynamic_auc(train_X_structured, test_X_structured, np.exp(pred_agg_risks_df['Predicted_risk']), times)

    coxloss = coxph_loss(torch.tensor(pred_agg_risks_df['Predicted_risk']), torch.tensor(target_agg_surv_df['Survival']), torch.tensor(target_agg_surv_df['Censor']))
    cindex = c_index(torch.tensor(pred_agg_risks_df['Predicted_risk']), torch.tensor(target_agg_surv_df['Survival']), torch.tensor(target_agg_surv_df['Censor']))
    
    test_KM_df = pd.merge(pred_agg_risks_df, target_agg_surv_df, left_index=True, right_index=True, how='inner')
    
    print('Test coxloss: {:.4f}; cindex: {:.4f}; auc3y: {:.4f}'.format(coxloss, cindex, test_aucs[2]))
    return  cindex, test_aucs[2], test_KM_df

def summary_score(fold,model, loader, n_classes):
    results_dir = n_classes.results_dir
    n_classes = n_classes.n_classes 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['HE_entity_submitter_id']

    patient_results = {}
    WSI_logits = []
    slide_time = []
    slide_event = []

    for batch_idx, batch in enumerate(loader):
        if hasattr(model, "num_clusters"):
            data, cluster_id, label = batch
            # data, cluster_id, label = data.to(device), cluster_id, label.to(device)
        else:
            data, label,y_time,y_event  = batch
            data, label = data.to(device), label.to(device)
            cluster_id = None
        #data, label = data.to(device), label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data, cluster_id=cluster_id)
            #logits, Y_prob, Y_hat, _, _ = model(data)
            WSI_logits.append(logits[0][0].cpu())
            slide_time.append(y_time[0].item())
            slide_event.append(y_event[0].item())
    # print('==============',logits[0][0])
    final_score = pd.DataFrame({'HE_entity_submitter_id': slide_ids,  'val_score' : np.array(WSI_logits)})
    WSI_logits = torch.tensor(WSI_logits)
    slide_time = torch.tensor(slide_time)
    slide_event = torch.tensor(slide_event)

    coxloss = coxph_loss(WSI_logits,slide_time,slide_event)
    cindex = c_index(WSI_logits,slide_time,slide_event)
    print('Test coxloss: {:.4f}; cindex: {:.4f}'.format(coxloss, cindex))
    final_score.to_csv(os.path.join(results_dir,'fold_'+str(fold)+'.csv' ))
    return  cindex
