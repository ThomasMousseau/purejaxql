import hydra
from omegaconf import OmegaConf

from purejaxql.distributional_pqn_rnn_craftax import single_run, tune


def _run_with_variant(config: dict, variant: str) -> None:
    config["ALG_VARIANT"] = variant
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    _run_with_variant(config, variant="qtd")


if __name__ == "__main__":
    main()
