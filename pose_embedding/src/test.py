import os

import hydra
from lightning import seed_everything


@hydra.main(version_base="1.3", config_path="../configs")
def main(cfg):
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_MODE"] = "disabled"
    seed_everything(cfg.general.seed, cfg.general.seed_workes)

    model = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.data)

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.logger.log_hyperparams(cfg)
    trainer.logger.watch(model)

    trainer.test(
        model,
        datamodule=datamodule,
        # ckpt_path=PATH_TO_CHECKPOINT,
    )

    trainer.logger.finalize("success")


if __name__ == "__main__":
    main()
