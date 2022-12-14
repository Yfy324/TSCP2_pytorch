
import os
import numpy as np
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict
# for typing!
import torch
from torch.functional import F
from torch.utils.data import Dataset
from torch.nn import Module
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, 
                train_set:Dataset, 
                val_set:Dataset,
                test_set:Dataset, 
                model: Module,
                optimizer: optim, 
                scheduler: lr_scheduler, 
                loss_fn:Module,
                writer:SummaryWriter,
                save_path:str,
                num_classes:int = 1,
                patience:int= 10, 
                device='cuda:0',
                 **kwargs):
        """Contrastive Learning Trainer

        Args:
            train_set (Dataset): _description_
            val_set (Dataset): _description_
            test_set (Dataset): _description_
            model (Module): _description_
            optimizer (optim): _description_
            scheduler (lr_scheduler): _description_
            loss_fn (Module): _description_
            writer (SummaryWriter): _description_
            save_path (str): _description_
            num_classes (int, optional): _description_. Defaults to 1.
            patience (int, optional): _description_. Defaults to 10.
            device (str, optional): _description_. Defaults to 'cuda:0'.
        """
        self.__dict__ = kwargs
        self.patience = 0 
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.writer = writer
        self.save_path = save_path
        self.num_classes = num_classes
        self.patience = patience
        self.device = device if torch.cuda.is_available() else 'cpu'
        # Metrics
        
        self.train_log = defaultdict(list,  # 几乎同dict，但可避免KeyError异常，命名key+设定values
                        { k:[] for k in ('train_loss', 'train_neg', 'train_sim') })
        self.val_log = defaultdict(list,
                        { k:[] for k in ('val_loss', 'val_neg', 'val_sim') })
        self.test_log = defaultdict(list,
                        { k:[] for k in ('loss', 'sim', 'y_gt') })
        
    def train_loop(self, epochs:int, 
                    win_sz:int,
                    temp=0.1,
                    beta:int= 1,
                    tau:float= 0.1)->Tuple[Dict, Dict]:
        """Training and validation steps."""
        
        best_val_loss = np.inf
        patience = 0
        
        # Epochs
        tbar = tqdm(range(epochs), position=0)
        
        for epoch in tbar:
            # Steps
            train_loss, train_sim, train_neg = self.train_step(epoch, win_sz, temp)
            val_loss, val_sim, val_neg = self.val_step(epoch)
            msg = (f"Train_sim:{train_sim:.3f}/neg:{train_neg:.3f}, "
                    f"Val_sim:{val_sim:.3f}/neg:{val_neg:.3f}")

            tbar.set_description(f"{msg}, Epoch: {epoch} |"
                                f" Train_loss: {train_loss:.3f},"
                                f" Val_loss: {val_loss:.3f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0 # reset patience
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience += 1
            if self.patience < patience: # 0
                print ("Stopping early!")
                break
        return self.train_log, self.val_log

    def train_step(self, 
                    epoch,
                    win_sz,
                    temperature=0.1):
        """Training one epoch."""
        # Set model to train mode
        self.model.train()
        # Logging
        log = defaultdict(float, { k:0.0 for k in ('epoch', 'loss', 'sim', 'loss') })
        
        # Iterate over train batches
        for i, (X, _) in enumerate(self.train_set.generate_batches()):
            # Split X1=X[0], X2=[X1]
            # Don't Use Label!
            X1 = X[0]
            X2 = X[1]
            
            # Set device
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            
            # Forward pass
            X1 = self.model(X1)
            X2 = self.model(X2)
            
            # X1 = F.normalize(X1, dim=1)
            # X2 = F.normalize(X2, dim=1)
            loss, sim_mean, sim_neg = self.loss_fn(X1, X2)
            
            # Backward pass + optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update batch metrics
            log['sim'] = (sim_mean.detach().cpu().numpy()+ i*log['sim'])/(i+1)
            log['neg'] = (sim_neg.detach().cpu().numpy() + i*log['neg'])/(i+1)
            log['loss'] = (loss.detach().cpu().numpy().mean()+ i*log['loss'])/(i+1)
            
            # Adjust learning rate
            self.scheduler.step(optimizer= self.optimizer, 
                                i= i,
                                epoch=epoch)
        for k, v in log.items():
            self.train_log[f'train_{k}'].append(np.mean(v))
            # Write to TensorBoard
            self.writer.add_scalar(tag=f'train_{k}', scalar_value=v, global_step=epoch)
        
        return log['loss'], log['sim'], log['neg']

    def val_step(self, epoch):
        """Validate one epoch."""
        # Set model to eval mode
        self.model.eval()
        # Logging
        log = defaultdict(float, { k:0.0 for k in ('loss', 'sim', 'neg') })
        
        # Iterate over val batches
        for i, (X, _) in enumerate(self.val_set.generate_batches()):
            # Split X1=X[0], X2=[X1]
            # Don't Use Label!
            X1 = X[0]
            X2 = X[1]
            # Set device
            # Set device
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            
            
            # Forward pass
            with torch.no_grad():
                # Forward pass
                X1 = self.model(X1)
                X2 = self.model(X2)
                # X1 = F.normalize(X1, dim=1)
                # X2 = F.normalize(X2, dim=1)
                loss, sim_mean, sim_neg = self.loss_fn(X1, X2)
                
            # Update batch metrics  -- moving average
            log['sim']= (sim_mean.detach().cpu().numpy()+ i*log['sim']) / (i+1)
            log['neg']= (sim_neg.detach().cpu().numpy()+ i*log['neg']) / (i+1)
            log['loss']= (loss.detach().cpu().numpy().mean()+ i*log['loss']) / (i+1)
        
        
        for k, v in log.items():
            self.val_log[f'val_{k}'] = np.mean(v)
            # Write to TensorBoard
            self.writer.add_scalar(tag=f'val_{k}', scalar_value=v, global_step=epoch)
        
        return log['loss'], log['sim'], log['neg']

    def test_loop(self,
                outpath:str,
                threshold:float= 0.5,
                metric:str= 'cosine')->Tuple[Dict, List, List]:
        self.model.eval()
        
        """Evalution of the test set."""
        # Logging
        log = defaultdict(list, { k:[] for k in ('epoch', 'similarity', 'metric') })

        # Iterate over val batches
        for i, (X, y) in enumerate(self.test_set.generate_batches()):
            # Split X1=X[0], X2=[X1]
            # Use Label!
            X1 = X[0]
            X2 = X[1]
            # Set device
            
            X1 = X1.to(self.device) #history
            X2 = X2.to(self.device) #future
            
            
            # Forward pass
            with torch.no_grad():
                # Forward pass
                X1 = self.model(X1)
                X2 = self.model(X2)
                X1 = F.normalize(X1, dim=1)
                X2 = F.normalize(X2, dim=1)
                sim = F.cosine_similarity(torch.unsqueeze(X1, dim=1),
                                        torch.unsqueeze(X2, dim=0), 
                                        dim=1)
            log['epoch'].append(i)
            log['metric'].append(metric)
            log['similarity'].append(sim.detach().cpu().numpy().tolist())

        print("Saved test similarity result!")
        import json
        with open(outpath, 'w') as f:
            json.dump(log, f, indent=4)
        
        return np.array(log['similarity'])