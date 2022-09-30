# from common import *
# from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.cli import LightningCLI

# from prover.datamodule import ProofDataModule
# from prover.model import EntailmentWriter
from model import RuleTakerModel
from dataloader import RuleTakerDataModule
from typing import Any


class RuletakerCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: Any) -> None:
        parser.link_arguments("model.plm", "data.plm", apply_on="instantiate")


def main() -> None:
    cli = RuletakerCLI(
        RuleTakerModel,
        RuleTakerDataModule,
        save_config_overwrite=True,
        auto_registry=True,
    )
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
