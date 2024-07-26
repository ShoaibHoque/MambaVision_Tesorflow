""" Plateau Scheduler

Adapts PyTorch plateau scheduler and allows application of noise, warmup.

Hacked together by / Copyright 2020 Ross Wightman
"""
import tensorflow as tf
import numpy as np

from .scheduler import Scheduler


class PlateauLRScheduler(Scheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self,
                 optimizer: tf.keras.optimizers.Optimizer,
                 decay_rate=0.1,
                 patience_t=10,
                 verbose=True,
                 threshold=1e-4,
                 cooldown_t=0,
                 warmup_t=0,
                 warmup_lr_init=0,
                 lr_min=0,
                 mode='max',
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize=True,
                 ):
        super().__init__(
            optimizer,
            'learning_rate',
            noise_range_t=noise_range_t,
            noise_type=noise_type,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=decay_rate, 
            patience=patience_t, 
            verbose=verbose, 
            mode=mode, 
            min_delta=threshold, 
            cooldown=cooldown_t, 
            min_lr=lr_min
        )

        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        self.restore_lr = None

    def state_dict(self):
        return {
            'best': self.lr_scheduler.best,
            'last_epoch': self.lr_scheduler._last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.lr_scheduler.best = state_dict['best']
        if 'last_epoch' in state_dict:
            self.lr_scheduler._last_epoch = state_dict['last_epoch']

    def step(self, epoch, metric=None):
        if epoch <= self.warmup_t:
            lrs = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
            super().update_groups(lrs)
        else:
            if self.restore_lr is not None:
                # restore actual LR from before our last noise perturbation before stepping base
                self.optimizer.learning_rate.assign(self.restore_lr)
                self.restore_lr = None

            self.lr_scheduler.on_epoch_end(epoch, logs={'val_loss': metric})

            if self._is_apply_noise(epoch):
                self._apply_noise(epoch)

    def _apply_noise(self, epoch):
        noise = self._calculate_noise(epoch)

        # apply the noise on top of previous LR, cache the old value so we can restore for normal
        # stepping of base scheduler
        old_lr = float(self.optimizer.learning_rate.numpy())
        new_lr = old_lr + old_lr * noise
        self.restore_lr = old_lr
        self.optimizer.learning_rate.assign(new_lr)
