from higan_contrastive.trainer import HiGANTrainer


def test_max_steps_zero_or_none_is_unlimited():
    trainer = object.__new__(HiGANTrainer)
    trainer.global_step = 100
    trainer.max_steps = None
    assert not trainer._step_limit_reached()


def test_max_steps_uses_resumed_global_step():
    trainer = object.__new__(HiGANTrainer)
    trainer.max_steps = 100
    trainer.global_step = 99
    assert not trainer._step_limit_reached()
    trainer.global_step = 100
    assert trainer._step_limit_reached()
