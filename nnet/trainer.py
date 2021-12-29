
import os
import subprocess
from typing import Tuple

import torch as th
from torch.nn.modules.loss import BCEWithLogitsLoss
from tqdm import tqdm

from .utils import StrLabelConverter, get_logger

import pdb

class Trainer(object):
    def __init__(self, model:th.nn.Module, checkpoint:str, tensorboard_dir:str, comment:str,  alphabet:str, learning_rate:float, verbose:bool, resume_path:str=None):
        self._verbose = verbose
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)
        if self._verbose: 
            print(f"Trainer initializing...\nUsing experiment directory: {self.checkpoint}")
        self.logger.info(f"Trainer initializing...")
        self.logger.info(f"Using experiment directory: {self.checkpoint}")

        if th.cuda.is_available():
            #Nvidia smi call
            freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes"| grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
            if len(freeGpu) == 0: # if gpu not aviable use cpu
                raise RuntimeError("CUDA device unavailable...exist")
            self._device = th.device('cuda:'+freeGpu.decode().strip())
            self.gpuid = (int(freeGpu.decode().strip()), )
        else:
            self._device = th.device("cpu")
            self.gpuid = None
        if self._verbose: print(f"Use device: {self._device}")
        self.logger.info(f"Use device: {self._device}")

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir = tensorboard_dir+'_'+comment)
        if self._verbose: print(f"Tensorboard is set as: {tensorboard_dir+'_'+comment}")
        self.logger.info(f"Tensorboard is set as: {tensorboard_dir+'_'+comment}")


        self._cur_epoch = 0
        self._converter = StrLabelConverter(alphabet)

        if resume_path is not None:
            self._cur_epoch, self._model, self._optimizer = self._load_checkpoint(resume_path, model, th.optim.Adam(), learning_rate)
            if self._verbose: print(f"Model and optimizer loaded from resume path: {resume_path}. LR: {learning_rate}")
            self.logger.info(f"Model and optimizer loaded from resume path: {resume_path}. LR: {learning_rate}")
        else:
            self._model = model
            self._optimizer = th.optim.Adam(self._model.parameters(), lr=learning_rate)
            if self._verbose: print(f"Model and optimizer set. LR: {learning_rate}")
            self.logger.info(f"Model and optimizer set. LR: {learning_rate}")

        self._model = self._model.to(self._device)
        self._loss_fn = th.nn.CTCLoss(zero_infinity=True)
        if self._verbose: print(f"Use loss_fn: CTCLoss\nComplete")
        self.logger.info(f"Use loss_fn: CTCLoss")
        self.logger.info(f"Complete")



    def _load_checkpoint(self, resume:str, model:th.nn.Module, optimizer:str, optimizer_kwargs)->Tuple[int, th.nn.Module, th.optim.Optimizer]:
        if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
        cpt = th.load(resume, map_location="cpu")
        cur_epoch = cpt["epoch"]
        self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
            resume, cur_epoch))
        # load nnet
        model.load_state_dict(cpt["model_state_dict"])
        return cur_epoch, model, optimizer(optimizer_kwargs, state=cpt["optim_state_dict"])

    def _save_checkpoint(self, cur_epoch:int, model:th.nn.Module, optimizer:th.optim, best:bool=True, apendix:str = ""):
        """Saves current state of given model and optimizer

        Args:
            cur_epoch (int): [description]
            model (th.nn.Module): [description]
            optimizer (th.optim): [description]
            best (bool, optional): [description]. Defaults to True.
            apendix (str, optional): [description]. Defaults to "".
        """        
        cpt = {
            "epoch": cur_epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best{}".format(apendix) if best else "last{}".format(apendix))))

    def train(self, dataloader):
        if self._verbose: print(f"Train")
        self.logger.info(f"Train")
        self._model.train()
        
        size = len(dataloader)
        sum_loss = 0
        
        for X, y in tqdm(dataloader) if self._verbose else dataloader:
            X = X.to(self._device)
           
            #y.to(self._device)
            # Compute prediction and loss
            pred = self._model(X)
            t, l = self._converter.encode(y)
            self._optimizer.zero_grad()
            batch_size = X.shape[0]
            preds_size = th.LongTensor([pred.shape[0]] * batch_size)
            loss = self._loss_fn(pred, t, preds_size, l) / batch_size
            sum_loss += loss
            # Backpropagation
            loss.backward()
            self._optimizer.step()
        return sum_loss/size
            
    def validate(self, dataloader):
        if self._verbose: print(f"Validation")
        self.logger.info(f"Validation")
        self._model.eval()

        val_string_targets = []
        predicted_strings_all = []
        
        size = len(dataloader)
        sum_loss = 0

        with th.no_grad():
            for X, y in tqdm(dataloader) if self._verbose else dataloader:
                X = X.to(self._device)
                #y.to(self._device)
                pred = self._model(X)
                t, l = self._converter.encode(y)
                batch_size = X.shape[0]
                preds_size = th.LongTensor([pred.shape[0]] * batch_size)
                loss = self._loss_fn(pred, t, preds_size, l) / batch_size
                sum_loss += loss

                _, preds = pred.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                decoded_strings = self._converter.decode(preds, preds_size)

                predicted_strings_all.extend(decoded_strings)
                val_string_targets.extend(y)


        #word_error_rate = getWer(predicted_strings_all, val_string_targets)

        #print(f"Word Error Rate: {word_error_rate}")
        #print(f"Character Error Rate: {char_error_rate}")
        return loss/size

    def _run(self, train_loader, dev_loader, num_epochs=50):
        # avoid alloc memory from gpu0
        # Check if save is OK
        self._save_checkpoint(cur_epoch=self._cur_epoch, model=self._model, optimizer=self._optimizer, best=False)
        # Base validation
        best_loss = self.validate(dev_loader)
        if self._verbose:
            print("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
            self._cur_epoch, best_loss))

        self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
            self._cur_epoch, best_loss))

        # Run epoch
        while self._cur_epoch < num_epochs:
            # Increase epoch_counter
            self._cur_epoch += 1
            if self._verbose: print(f"Epoch: {self._cur_epoch}")
            # Train
            tr = self.train(train_loader)
            # Validate
            if self._cur_epoch % 10 == 0:
                cv = self.validate(dev_loader)
            else:
                cv = None

            if self._verbose:
                if cv:
                    print(f"Train_loss: {tr} | CV_loss: {cv}")
                else:
                    print(f"Train_loss: {tr}")

            # Log values
            if cv:
                self.logger.info(f"Train_loss: {tr} | CV_loss: {cv}")
            else:
                self.logger.info(f"Train_loss: {tr}")
            #Tensorboard
            self._writer.add_scalar("OCR/Train", tr, self._cur_epoch)
            if cv:
                self._writer.add_scalar("OCR/CrossValidation", cv, self._cur_epoch)
            self._writer.flush()                    
            if cv:
                if cv < best_loss:
                    best_loss = cv
                    self._save_checkpoint(cur_epoch=self._cur_epoch, model=self._model, optimizer=self._optimizer, best=True)
            
            self._save_checkpoint(cur_epoch=self._cur_epoch, model=self._model, optimizer=self._optimizer, best=False)

    def run(self, train_loader, dev_loader, num_epochs=50):
        if self._verbose: print(f"Use device: {self._device}")
        self.logger.info(f"Use device: {self._device}")
        # avoid alloc memory from gpu0
        
        if self.gpuid:
            print("DBG")
            with th.cuda.device(self.gpuid[0]):
                self._run(train_loader=train_loader, dev_loader=dev_loader, num_epochs=num_epochs)
        else:
            self._run(train_loader=train_loader, dev_loader=dev_loader, num_epochs=num_epochs)
        
        self.logger.info("Training for {:d}/{:d} epoches done!".format(
            self._cur_epoch, num_epochs))
