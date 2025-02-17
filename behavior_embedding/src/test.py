import os

import hydra
import omegaconf
from lightning import seed_everything


@hydra.main(version_base="1.3", config_path="../configs")
def main(cfg):
    seed_everything(cfg.general.seed, cfg.general.seed_workers)

    os.environ["WANDB_MODE"] = "offline"

    model = hydra.utils.instantiate(cfg.model)
    print(model)
    datamodule = hydra.utils.instantiate(cfg.data)

    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.logger._log_graph = True
    cfg_dict = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    trainer.logger.log_hyperparams(cfg_dict)
    trainer.test(
        model,
        datamodule=datamodule,
        # ckpt_path=PATH_TO_CHECKPOINT,
    )

    trainer.logger.finalize("success")


if __name__ == "__main__":
    main()
