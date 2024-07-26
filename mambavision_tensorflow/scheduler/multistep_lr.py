import bisect
import tensorflow as tf

class MultiStepLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    MultiStep LR Scheduler with warmup and optional noise.
    """

    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 decay_rate=1.0,
                 warmup_steps=0,
                 warmup_lr_init=0.0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42):
        super(MultiStepLRScheduler, self).__init__()
        
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_std = noise_std
        self.noise_seed = noise_seed

        if self.warmup_steps:
            self.warmup_steps_values = [(self.initial_learning_rate - warmup_lr_init) / self.warmup_steps]
        else:
            self.warmup_steps_values = [1]

    def get_curr_decay_steps(self, step):
        # find where in the array step goes, assumes self.decay_steps is sorted
        return bisect.bisect_right(self.decay_steps, step + 1)

    def _get_lr(self, step):
        if step < self.warmup_steps:
            lrs = [self.warmup_lr_init + step * s for s in self.warmup_steps_values]
        else:
            lrs = [self.initial_learning_rate * (self.decay_rate ** self.get_curr_decay_steps(step))]
        return lrs[0]

    def __call__(self, step):
        return self._get_lr(step)

    def get_config(self):
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'decay_rate': self.decay_rate,
            'warmup_steps': self.warmup_steps,
            'warmup_lr_init': self.warmup_lr_init,
            't_in_epochs': self.t_in_epochs,
            'noise_range_t': self.noise_range_t,
            'noise_pct': self.noise_pct,
            'noise_std': self.noise_std,
            'noise_seed': self.noise_seed,
        }
        return config
