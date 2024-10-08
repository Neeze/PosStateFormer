import zipfile
from typing import List , Tuple
import time
import math
import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor

from HybridFormer.datamodule import Batch, vocab ,label_make_muti
from HybridFormer.model.hybridformer import HybridFormer
from HybridFormer.utils.utils import (
    ExpRateRecorder, 
    Hypothesis, 
    ce_loss_all, 
    ce_loss, 
    to_bi_tgt_out
)

class LitPosFormer(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        d_state: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        warmup_steps: int,
        learning_rate: float,
        min_learning_rate:float,
        gamma:float,
        patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = HybridFormer(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            d_state=d_state,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, logger
    ) -> Tuple[FloatTensor,FloatTensor]:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.model(img, img_mask, tgt, logger)

    def training_step(self, batch: Batch, _):      
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat , out_hat_layer ,out_hat_pos = self(batch.imgs, batch.mask, tgt , self.trainer.logger)
        tgt_list=tgt.cpu().numpy().tolist()
        layer_num , final_pos=label_make_muti.out2layernum_and_pos(tgt_list)
        layer_num_tensor=torch.LongTensor(layer_num)   #[2b,l,5]
        final_pos_tensor=torch.LongTensor(final_pos)   #[2b,l,6]
        layer_num_tensor=layer_num_tensor.cuda()
        final_pos_tensor=final_pos_tensor.cuda()  
        loss, layer_loss, pos_loss  = ce_loss_all(out_hat, out,out_hat_layer,layer_num_tensor,out_hat_pos,final_pos_tensor)
        self.log("train_loss", loss, logger=True, on_step=False, on_epoch=True, sync_dist=True,prog_bar=True)
        self.log("train_loss_pos",pos_loss, logger=True, on_step=False, on_epoch=True,prog_bar=True, sync_dist=True)
        self.log("train_loss_layernum",layer_loss, logger=True,on_step=False, on_epoch=True, sync_dist=True,prog_bar=True)
        loss = (loss+0.25*layer_loss+0.25*pos_loss)/1.5
        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat , out_hat_layer ,out_hat_pos = self(batch.imgs, batch.mask, tgt,self.trainer.logger)
        
        tgt_list=tgt.cpu().numpy().tolist()
        layer_num , final_pos=label_make_muti.out2layernum_and_pos(tgt_list)
        layer_num_tensor=torch.LongTensor(layer_num)   #[2b,l,5]
        final_pos_tensor=torch.LongTensor(final_pos)   #[2b,l,6]
        layer_num_tensor=layer_num_tensor.cuda()
        final_pos_tensor=final_pos_tensor.cuda()  
        
        loss,layer_loss,pos_loss  = ce_loss_all(out_hat, out,out_hat_layer,layer_num_tensor,out_hat_pos,final_pos_tensor)

        self.log(
            "val_loss",
            loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_loss_pos",
            pos_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_loss_layernum",
            layer_loss,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            logger=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Batch, _):
        start_time = time.time()  # Start timing
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        inference_time = time.time() - start_time  # Compute inference time for this batch
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log('batch_inference_time', inference_time)  # Optional: log inference time per batch
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps], inference_time

    def test_epoch_end(self, test_outputs) -> None:
        total_inference_time = sum(output[2] for output in test_outputs)  # Sum up the inference times
        print(f"Total Inference Time: {total_inference_time} seconds")

        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds, _ in test_outputs:  # Unpack the ignored time measurements
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        # optimizer = optim.SGD(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     momentum=0.9,
        #     weight_decay=1e-4,
        # )

        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
        )


        # reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="max",
        #     factor=0.25,
        #     patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        # )

        # reduce_scheduler = self.cosine_scheduler(
        #     optimizer,
        #     training_steps=self.trainer.max_steps,
        #     warmup_steps=self.hparams.warmup_steps,
        # )

        # scheduler = {
        #     "scheduler": reduce_scheduler,
        #     "monitor": "val_ExpRate",
        #     "interval": "epoch",
        #     "frequency": self.trainer.check_val_every_n_epoch,
        #     "strict": True,
        # }

        reduce_scheduler = self.exponential_scheduler(
                optimizer,
                self.hparams.warmup_steps,
                self.hparams.learning_rate,
                self.hparams.min_learning_rate,
                self.hparams.gamma,
            )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "step",
            "frequency": 20,
            "strict": True,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def exponential_scheduler(optimizer, warmup_steps, lr, min_lr=5e-5, gamma=0.9999):
        def lr_lambda(x):
            if x > warmup_steps or warmup_steps <= 0:
                if lr * gamma ** (x - warmup_steps) > min_lr:
                    return gamma ** (x - warmup_steps)
                else:
                    return min_lr / lr
            else:
                return x / warmup_steps

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
