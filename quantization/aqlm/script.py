from .config import AQLMConfig, CalibrationConfig

from dataclasses import dataclass, field, asdict
from abc import abstractmethod
from pathlib import Path
from typing import Optional
import subprocess

import logging

logger = logging.getLogger(__name__)


@dataclass
class Script:
    script_name: str = field(init=False)

    _script_root: Path = Path(__file__).parent / "scripts"
    _script: Optional[str] = field(init=False, default=None)
    _script_path: Optional[Path] = field(init=False, default=None)

    @property
    def script(self):
        if self._script is None:
            raise ValueError("Script not built yet.")
        return self._script

    @property
    def script_path(self):
        if self._script_path is None:
            raise ValueError("Script not saved yet.")
        return self._script_path

    def __post_init__(self):
        assert (
            self._script_root.exists()
        ), f"Script root {self._script_root} does not exist."
        assert (
            self._script_root.is_dir()
        ), f"Script root {self._script_root} is not a directory."
        assert (
            self._script_root / self.script_name
        ).exists(), f"Script {self.script_name} does not exist in {self._script_root}."

    @abstractmethod
    def get_config_dict(self, config: AQLMConfig) -> dict:
        """Get the config dictionary from the AQLMConfig object"""
        ...

    def build_script(self, config: AQLMConfig) -> "Script":
        # load the default script and replace the placeholders with the config values
        if self._script is not None:
            logger.warning("Script is already built. Skipping build.")

        with open(self._script_root / self.script_name, "r") as f:
            script = f.read()
        config_dict = self.get_config_dict(config)
        script = script.format(**config_dict)
        self._script = script
        return self

    def save_script(self, path: Path) -> None:
        with open(path, "w") as f:
            f.write(self.script)
        self._script_path = path

    def run_script(self) -> None:
        subprocess.check_call(["bash", self.script_path])


@dataclass
class CalibrationScript(Script):
    script_name: str = "calibrate.sh"

    def get_config_dict(self, config: AQLMConfig) -> dict:
        config_dict = asdict(config.calibration_config)
        config_dict.update({
            "model_path": config.model_path,
            "output_path": config.output_path,
        })
        return 
