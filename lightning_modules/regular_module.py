import os
from tqdm import tqdm
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchmetrics.text import WordErrorRate

from utils.utils import initialize_config
from utils.text_process import TextTransform
from models.decoder import get_beam_decoder

class BaseASRModule(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.ASR_model = initialize_config(config["ASR_model"])
        self.greedy_decoder = initialize_config(config["decoder"]["greedy_decoder"])
        token_file = os.path.join(
            self.config["validation_dataset"]["args"]["data_dir"],
            self.config["val_dataloader"]["collate_fn"]["args"]["token_file"]
        )
        self.text_transform = TextTransform(token_file=token_file)
        self.ter = WordErrorRate()

        self.val_ter_outputs = []
        self.val_acc_outputs = []

    def _compute_loss(self, waveforms, labels, input_lengths, label_lengths, step_type):
        """
        Compute the loss for a given batch of noisy and clean waveforms.

        Args:
            waveforms: The waveforms input
            labels: The labels of the data
            input_lengths: List of lengths of waveforms 
            label_lengths: List of lenghts of labels
            step_type: The step type, either 'train' or 'val' for logging purposes.

        Returns:
            The computed loss for the current batch.
        """
        loss, log_statistics = self.ASR_model(waveforms, labels, input_lengths, label_lengths)
        self._log_statistics(loss, log_statistics, step_type)
        return loss

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
        waveforms, labels, input_lengths, label_lengths, references, references_word = batch
        return self._compute_loss(waveforms, labels, input_lengths, label_lengths, "train")

    def validation_step(self, batch, batch_idx):
        waveforms, labels, input_lengths, label_lengths, references, references_word = batch
        loss = self._compute_loss(waveforms, labels, input_lengths, label_lengths, "val")
        outputs = self.ASR_model.inference(waveforms)

        beam_search_decoder = get_beam_decoder(
            split='validation',
            config=self.config
        )
        pred_seqs, pred_words = self._predict(outputs, beam_search_decoder)
        # print (pred_words)

        for i in range(len(pred_words)):
            self.val_ter_outputs.append([pred_seqs[i], references[i]])
            self.val_acc_outputs.append([pred_words[i], references_word[i]])
        return loss

    def on_validation_epoch_end(self):
        all_ter_preds, all_ter_labels = \
            [output[0] for output in self.val_ter_outputs], [output[1] for output in self.val_ter_outputs]
        all_acc_preds, all_acc_labels = \
            [output[0] for output in self.val_acc_outputs], [output[1] for output in self.val_acc_outputs]
        self.log("val/ter", self.ter(all_ter_preds, all_ter_labels), sync_dist=True)
        self.log("val/acc", self.compute_acc(all_acc_preds, all_acc_labels), sync_dist=True)
        self.val_ter_outputs.clear()
        self.val_acc_outputs.clear()

    def _predict(self, outputs, beam_search_decoder):
        inds = self.greedy_decoder(outputs.detach())
        pred_seqs = [self.text_transform.int_to_text(ind) for ind in inds]
        # if pred_seq in word_clause:
        #     pred_word = word_clause[pred_seq]
                
        # else:
        beam_search_result = beam_search_decoder(outputs.detach().cpu())
        pred_words = [" ".join(hypo[0].words).strip() for hypo in beam_search_result]
        # j = 1
        # while pred_word == '' and j < len(beam_search_result[0]):
        #     pred_word = " ".join(beam_search_result[0][j].words).strip()
        #     j += 1

        return pred_seqs, pred_words

    def inference(self, dataloader, config):
        """
        Evaluate model on test dataest.

        Args:
            dataloader: test dataloader
        """
        test_ter_outputs, test_acc_outputs = [], []
        beam_search_decoder = get_beam_decoder(
            split='test',
            config=config
        )

        j = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                waveforms, labels, input_lengths, label_lengths, references, references_word = batch
                waveforms = waveforms.to(device=self.device)
                outputs = self.ASR_model.inference(waveforms)
        
                pred_seqs, pred_words = self._predict(outputs, beam_search_decoder)
        
                for i in range(len(pred_words)):
                    test_ter_outputs.append([pred_seqs[i], references[i]])
                    test_acc_outputs.append([pred_words[i], references_word[i]])
                
                j += 1

        return test_ter_outputs, test_acc_outputs
    

    def compute_acc(self, preds, labels):
        return sum(np.array(preds) == np.array(labels)) / len(preds)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.ASR_model.parameters(),
            lr=self.config["optimizer"]["lr"],
            weight_decay=self.config["optimizer"]["weight_decay"],
            betas=(
                self.config["optimizer"]["beta1"],
                self.config["optimizer"]["beta2"],
            ),
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', 
            threshold=self.config["lr_scheduler"]["threshold"], 
            factor=self.config["lr_scheduler"]["factor"], 
            patience=self.config["lr_scheduler"]["patience"],
            min_lr=self.config["lr_scheduler"]["min_lr"]
        )
        # return [optimizer], [lr_scheduler]
        return {
            'optimizer': optimizer,
            'scheduler': lr_scheduler,
            'monitor': 'val_acc'
        }

    def get_train_dataloader(self, trainset):
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