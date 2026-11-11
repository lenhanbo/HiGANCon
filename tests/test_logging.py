from higan_contrastive.trainer import HiGANTrainer


def test_log_window_keeps_sparse_generator_metrics():
    trainer = object.__new__(HiGANTrainer)
    trainer.log_window_sums = {}
    trainer.log_window_counts = {}
    trainer._update_log_window({"d_total": 2.0, "supcon": 1.0, "lambda_con": 0.02})
    trainer._update_log_window({"d_total": 4.0})
    averages = trainer._pop_log_window()
    assert averages["d_total"] == 3.0
    assert averages["supcon"] == 1.0
    assert averages["lambda_con"] == 0.02
    assert trainer.log_window_sums == {}
