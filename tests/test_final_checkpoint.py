from pathlib import Path
from unittest.mock import Mock

from skp.callbacks import FinalCheckpointCallback


def test_final_checkpoint_saves_latest_completed_epoch(tmp_path: Path):
    trainer = Mock()
    callback = FinalCheckpointCallback(tmp_path, save_weights_only=True)

    callback.on_train_epoch_end(trainer, Mock())

    trainer.save_checkpoint.assert_called_once_with(
        tmp_path / "last.ckpt",
        weights_only=True,
    )
