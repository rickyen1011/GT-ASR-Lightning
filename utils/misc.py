    # def _compute_loss(
    #     self, 
    #     waveforms, 
    #     labels,
    #     input_lengths, 
    #     label_lengths,
    #     labels_d, 
    #     step_type
    # ):
    #     """
    #     Compute the loss for a given batch of noisy and clean waveforms.

    #     Args:
    #         waveforms: The waveforms input
    #         labels: The labels of the data
    #         input_lengths: List of lengths of waveforms 
    #         label_lengths: List of lenghts of labels
    #         labels_d: The domain labels
    #         step_type: The step type, either 'train' or 'val' for logging purposes.

    #     Returns:
    #         The computed loss for the current batch.
    #     """
    #     (loss_asr, log_statistics), features = self.ASR_model(
    #         waveforms, labels, input_lengths, label_lengths
    #     )

    #     p = self.get_p()
    #     lambda_p = self.get_lambda_p(p)
    #     self.log("p", p, on_step=False, on_epoch=True, sync_dist=True)

    #     features_rev = GRL.apply(features, lambda_p)
    #     loss_d, _ = self.discriminator(features_rev, labels_d)

    #     loss = loss_asr + loss_d
    #     log_statistics['DiscriminatorLoss'] = loss_d.detach()

    #     self._log_statistics(loss, log_statistics, step_type)
    #     return loss