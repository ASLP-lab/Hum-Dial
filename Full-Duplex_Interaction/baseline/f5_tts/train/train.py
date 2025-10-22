# training script.

import os
from importlib.resources import files
import sys
import hydra
from omegaconf import OmegaConf
sys.path.append("/home/node60_tmpdata/hkxie/osum_dit/src")

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer


os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # 显式指定可见的GPU

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(cfg):
    tokenizer = cfg.model.tokenizer
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    exp_name = f"{cfg.model.name}_{mel_spec_type}"
    print(cfg.model)
    
    vocab_char_map, vocab_size = None,None
    
    # set model
    
    model_cls = DiT
    wandb_resume_id = None
    
    model = CFM(
        transformer=model_cls(**cfg.model.arch, text_num_embeds=cfg.model.text_num_embeds, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    
    # init trainer
    trainer = Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=getattr(cfg.ckpts, "keep_last_n_checkpoints", -1),
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{cfg.ckpts.save_dir}")),
        batch_size_per_gpu=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=cfg.ckpts.last_per_updates,
        log_samples=True,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=cfg.model.vocoder.is_local,
        local_vocoder_path=cfg.model.vocoder.local_path,
        feat = cfg.model.feat
    )
    # import pdb;pdb.set_trace()
    print(cfg.datasets.dataset_type)
    train_dataset = load_dataset(cfg.datasets.name, tokenizer, dataset_type=cfg.datasets.dataset_type ,mel_spec_kwargs=cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
