import pytorch_lightning as pl
from HybridFormer.datamodule import CROHMEDatamodule
from HybridFormer.lit_hybridformer import LitPosFormer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger as Logger
import argparse
from sconf import Config

def train(config):
    pl.seed_everything(config.seed_everything, workers=True)

    model_module = LitPosFormer(
        d_model = config.model.d_model,
        # encoder
        growth_rate = config.model.growth_rate,
        num_layers = config.model.num_layers,

        # decoder
        nhead = config.model.nhead,
        d_state = config.model.d_state,
        expand_factor = config.model.expand_factor,
        d_conv = config.model.d_conv,
        num_decoder_layers = config.model.num_decoder_layers,
        dim_feedforward = config.model.dim_feedforward,
        dropout = config.model.dropout,
        dc = config.model.dc,
        cross_coverage = config.model.cross_coverage,
        self_coverage = config.model.self_coverage,
        # beam search
        beam_size = config.model.beam_size,
        max_len = config.model.max_len,
        alpha = config.model.alpha,
        early_stopping = config.model.early_stopping,
        temperature = config.model.temperature,
        # training
        learning_rate = config.model.learning_rate,
        patience = config.model.patience,
    )
    data_module = CROHMEDatamodule(
        zipfile_path = config.data.zipfile_path,
        test_year = config.data.test_year,
        train_batch_size = config.data.train_batch_size,
        eval_batch_size = config.data.eval_batch_size,
        num_workers = config.data.num_workers,
        scale_aug = config.data.scale_aug,)
    
    logger = Logger("HybridFormer Project", project="hybridformer", config=dict(config), log_model='all')
    logger.watch(model_module.model, log="all", log_freq=100)

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=config.trainer.callbacks[1].init_args.save_top_k, 
                                                       monitor=config.trainer.callbacks[1].init_args.monitor, 
                                                       mode=config.trainer.callbacks[1].init_args.mode,
                                                       filename=config.trainer.callbacks[1].init_args.filename)
    
    trainer = pl.Trainer(
        devices=config.trainer.gpus,
        accelerator=config.trainer.accelerator,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        deterministic=config.trainer.deterministic,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        callbacks = [lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    train(config)