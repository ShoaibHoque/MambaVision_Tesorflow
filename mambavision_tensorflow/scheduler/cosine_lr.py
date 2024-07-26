import logging
import math
import numpy as np
import tensorflow as tf

_logger = logging.getLogger(__name__)

class CosineLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(self,
                 initial_learning_rate,
                 t_initial,
                 lr_min=0.,
                 cycle_mul=1.,
                 cycle_decay=1.,
                 cycle_limit=1,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 k_decay=1.0):
        super(CosineLRScheduler, self).__init__()

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.initial_learning_rate = initial_learning_rate
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.k_decay = k_decay
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed

        if self.warmup_t:
            self.warmup_steps = [(self.initial_learning_rate - warmup_lr_init) / self.warmup_t]
        else:
            self.warmup_steps = [1]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [self.initial_learning_rate * gamma]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in lr_max_values]

        return lrs

    def __call__(self, step):
        return self._get_lr(step)[0]

    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            't_initial': self.t_initial,
            'lr_min': self.lr_min,
            'cycle_mul': self.cycle_mul,
            'cycle_decay': self.cycle_decay,
            'cycle_limit': self.cycle_limit,
            'warmup_t': self.warmup_t,
            'warmup_lr_init': self.warmup_lr_init,
            'warmup_prefix': self.warmup_prefix,
            't_in_epochs': self.t_in_epochs,
            'noise_range_t': self.noise_range_t,
            'noise_pct': self.noise_pct,
            'noise_std': self.noise_std,
            'noise_seed': self.noise_seed,
            'k_decay': self.k_decay,
        }
        return config

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))
