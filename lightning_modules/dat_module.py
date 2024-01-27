import os, sys
from tqdm import tqdm
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from lightning_modules.regular_module import BaseASRModule
from torchmetrics import Accuracy
from torchmetrics.text import WordErrorRate

from utils.utils import initialize_config
from utils.text_process import TextTransform
from models.decoder import get_beam_decoder
from models.discriminator import GRL

class DATASRModule(BaseASRModule):
    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False
        self.config = config
        self.ASR_model = initialize_config(config["ASR_model"])
        self.discriminator = initialize_config(config["discriminator"])
        self.greedy_decoder = initialize_config(config["decoder"]["greedy_decoder"])
        token_file = os.path.join(
            self.config["validation_dataset"]["args"]["data_dir"],
            self.config["val_dataloader"]["collate_fn"]["args"]["token_file"]
        )
        self.text_transform = TextTransform(token_file=token_file)
        self.ter = WordErrorRate()

        self.val_ter_outputs = []
        self.val_acc_outputs = []
        self.val_dacc_outputs = []

    def _compute_d_loss(
        self, 
        waveforms, 
        labels,
        input_lengths, 
        label_lengths,
        labels_d, 
        step_type
    ):
        """
        Compute the loss for discriminator.

        Args:
            waveforms: The waveforms input
            labels: The labels of the data
            input_lengths: List of lengths of waveforms 
            label_lengths: List of lenghts of labels
            labels_d: The domain labels
            step_type: The step type, either 'train' or 'val' for logging purposes.

        Returns:
            The computed loss for the current batch.
        """
        (loss_asr, log_statistics), features = self.ASR_model(
            waveforms, labels, input_lengths, label_lengths
        )

        loss_d, _ = self.discriminator(features.detach(), labels_d)
        return loss_d


    def _compute_g_loss(
        self, 
        waveforms, 
        labels,
        input_lengths, 
        label_lengths,
        labels_d, 
        step_type
    ):
        """
        Compute the loss for ASR.

        Args:
            waveforms: The waveforms input
            labels: The labels of the data
            input_lengths: List of lengths of waveforms 
            label_lengths: List of lenghts of labels
            labels_d: The domain labels
            step_type: The step type, either 'train' or 'val' for logging purposes.

        Returns:
            The computed loss for the current batch.
        """
        (loss_asr, log_statistics), features = self.ASR_model(
            waveforms, labels, input_lengths, label_lengths
        )

        loss_d, _ = self.discriminator(features, labels_d)
        p = self.get_p()
        lambda_p = self.get_lambda_p(p)
        self.log("p", p, on_step=True, sync_dist=True)
        self.log("lambda_p", lambda_p, on_step=True, sync_dist=True)
        loss_g = loss_asr - lambda_p * loss_d
        
        log_statistics['DiscriminatorLoss'] = loss_d.detach()
        self._log_statistics(loss_g, log_statistics, step_type)
        return loss_g


    def _log_statistics(self, loss, log_statistics, step_type):
        """
        Log the statistics of the current training or validation step.

        Args:
            loss: The computed loss value to be logged.
            log_statistics: A dictionary of other statistics to be logged.
            step_type: The step type, either 'train' or 'val' for logging purposes.
        """
        self.log(
            f"{step_type}/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        for key, value in log_statistics.items():
            self.log(
                f"{step_type}/{key}", value, on_step=True, on_epoch=True, sync_dist=True
            )


    def training_step(self, batch, batch_idx):
        waveforms, labels, input_lengths, label_lengths, references, references_word, labels_d, langs = batch
        loss_d = self._compute_d_loss(
            waveforms, labels, input_lengths, label_lengths, labels_d, "train"
        )
        g_opt, d_opt = self.optimizers()
        d_opt.zero_grad()
        self.manual_backward(loss_d)
        d_opt.step()
        
        loss_g = self._compute_g_loss(
            waveforms, labels, input_lengths, label_lengths, labels_d, "train"
        )
        g_opt.zero_grad()
        d_opt.zero_grad()
        self.manual_backward(loss_g)
        g_opt.step()


    def validation_step(self, batch, batch_idx):
        waveforms, labels, input_lengths, label_lengths, references, references_word, labels_d, langs = batch
        outputs, features = self.ASR_model.inference(waveforms)
        outputs_d = self.discriminator.inference(features)
        outputs_d = torch.argmax(outputs_d, dim=-1)

        beam_search_decoder = get_beam_decoder(
            split='validation',
            model_config=self.config,
            test_config=self.config
        )
        pred_seqs, pred_words = self._predict(outputs, beam_search_decoder)
        # print (pred_words)

        for i in range(len(pred_words)):
            self.val_ter_outputs.append([pred_seqs[i], references[i]])
            self.val_acc_outputs.append([pred_words[i], references_word[i]])
            self.val_dacc_outputs.append([outputs_d.detach().cpu()[i], labels_d.cpu()[i]])


    def on_validation_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        all_ter_preds, all_ter_labels = \
            [output[0] for output in self.val_ter_outputs], [output[1] for output in self.val_ter_outputs]
        all_acc_preds, all_acc_labels = \
            [output[0] for output in self.val_acc_outputs], [output[1] for output in self.val_acc_outputs]
        all_dacc_preds, all_dacc_labels = \
            [output[0] for output in self.val_dacc_outputs], [output[1] for output in self.val_dacc_outputs]
        
        ter = self.ter(all_ter_preds, all_ter_labels)
        acc = self.compute_acc(all_acc_preds, all_acc_labels)
        sch_g.step(acc)
        sch_d.step()
        self.log(
            "val/ter", 
            ter, 
            on_step=False, 
            batch_size=self.config["val_dataloader"]["batch_size"], 
            sync_dist=True
        )
        self.log(
            "val/acc", 
            acc,
            on_step=False, 
            batch_size=self.config["val_dataloader"]["batch_size"],
            sync_dist=True
        )
        self.log("val/dacc", self.compute_acc(all_dacc_preds, all_dacc_labels), sync_dist=True)
        self.val_ter_outputs.clear()
        self.val_acc_outputs.clear()
        self.val_dacc_outputs.clear()


    def inference(self, dataloader, test_config):
        """
        Evaluate model on test dataest.

        Args:
            dataloader: test dataloader
        """
        test_ter_outputs, test_acc_outputs = {'All': []}, {'All': []}
        beam_search_decoder = get_beam_decoder(
            split='test',
            test_config=test_config,
            model_config=self.config,
        )

        j = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                waveforms, labels, input_lengths, label_lengths, references, references_word, labels_d, langs = batch
                waveforms = waveforms.to(device=self.device)
                outputs, features = self.ASR_model.inference(waveforms)
                outputs_d = self.discriminator.inference(features)
                outputs_d = torch.argmax(outputs_d, dim=-1)
        
                pred_seqs, pred_words = self._predict(outputs, beam_search_decoder)
        
                for i in range(len(pred_words)):
                    lang = langs[i]
                    if lang not in test_ter_outputs:
                        test_ter_outputs[lang] = []
                        test_acc_outputs[lang] = []
                    test_ter_outputs[lang].append([pred_seqs[i], references[i]])
                    test_acc_outputs[lang].append([pred_words[i], references_word[i]])
                    test_ter_outputs['All'].append([pred_seqs[i], references[i]])
                    test_acc_outputs['All'].append([pred_words[i], references_word[i]])
                
                j += 1

        return test_ter_outputs, test_acc_outputs


    def inference_dat(self, dataloader):
        """
        Analyze the domain invariant effect on test dataest.

        Args:
            dataloader: test dataloader
        """
        test_dacc_outputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                waveforms, labels, input_lengths, label_lengths, references, references_word, labels_d = batch
                waveforms = waveforms.to(device=self.device)
                outputs, features = self.ASR_model.inference(waveforms)
                outputs_d = self.discriminator.inference(features)
                outputs_d = torch.argmax(outputs_d, dim=-1)
        
                for i in range(len(outputs_d)):
                    test_dacc_outputs.append([outputs_d.detach().cpu()[i], labels_d.cpu()[i]])
                
        return test_dacc_outputs


    def configure_optimizers(self):
        optimizer_g = torch.optim.AdamW(
            params=self.ASR_model.parameters(),
            lr=self.config["optimizer_g"]["lr"],
            weight_decay=self.config["optimizer_g"]["weight_decay"],
            betas=(
                self.config["optimizer_g"]["beta1"],
                self.config["optimizer_g"]["beta2"],
            ),
        )
        optimizer_d = torch.optim.AdamW(
            params=self.discriminator.parameters(),
            lr=self.config["optimizer_d"]["lr"],
            betas=(
                self.config["optimizer_d"]["beta1"],
                self.config["optimizer_d"]["beta2"],
            ),
        )
        lr_scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, mode='max', 
            threshold=self.config["lr_scheduler_g"]["threshold"], 
            factor=self.config["lr_scheduler_g"]["factor"], 
            patience=self.config["lr_scheduler_g"]["patience"],
            min_lr=self.config["lr_scheduler_g"]["min_lr"]
        )
        lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_d, gamma=self.config["lr_scheduler_d"]["decay_factor"]
        )
        # return [optimizer], [lr_scheduler]
        return (
            {
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_g,
                    "monitor": "val/acc",
                },
            },
            {"optimizer": optimizer_d, "lr_scheduler": lr_scheduler_d},
        )


    def get_train_dataloader(self, trainset):
        self.len_dataloader = len(trainset) // self.config["train_dataloader"]["batch_size"]
        collate_fn = initialize_config(
            self.config["train_dataloader"]["collate_fn"],
        )
        return DataLoader(
            dataset=trainset,
            batch_size=self.config["train_dataloader"]["batch_size"],
            num_workers=self.config["train_dataloader"]["num_workers"],
            shuffle=self.config["train_dataloader"]["shuffle"],
            collate_fn=collate_fn.collate_fn,
            pin_memory=self.config["train_dataloader"]["pin_memory"],
        )


    def get_val_dataloader(self, valset):
        collate_fn = initialize_config(
            self.config["val_dataloader"]["collate_fn"],
        )
        return DataLoader(
            dataset=valset,
            batch_size=self.config["val_dataloader"]["batch_size"],
            num_workers=self.config["val_dataloader"]["num_workers"],
            collate_fn=collate_fn.collate_fn,
        )


    def get_test_dataloader(self, testset, test_config):
        collate_fn = initialize_config(
            test_config["test_dataloader"]["collate_fn"],
        )
        return DataLoader(
            dataset=testset,
            batch_size=test_config["test_dataloader"]["batch_size"],
            num_workers=test_config["test_dataloader"]["num_workers"],
            collate_fn=collate_fn.collate_fn,
        )


    def get_p(self):
        current_iterations = self.global_step
        current_epoch = self.current_epoch
        len_dataloader = self.len_dataloader
        p = float(current_iterations / (self.config["trainer"]["max_epochs"] * len_dataloader))
        return p


    def get_lambda_p(self, p):
        lambda_p = 2. / (1. + np.exp(-self.config["dat"]["gamma"] * p)) - 1
        return lambda_p