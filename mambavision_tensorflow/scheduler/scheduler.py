from typing import Dict, Any, List, Union
import tensorflow as tf
import numpy as np

class Scheduler:
    """ Parameter Scheduler Base Class for TensorFlow
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the built-in TensorFlow schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behavior. All epoch and update counts must be tracked in the training
    code and explicitly passed into the schedulers on the corresponding step or step_update call.
    """

    def __init__(self,
                 optimizer: tf.keras.optimizers.Optimizer,
                 param_group_field: str,
                 noise_range_t: Union[List[int], int] = None,
                 noise_type: str = 'normal',
                 noise_pct: float = 0.67,
                 noise_std: float = 1.0,
                 noise_seed: int = None,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        
        self.base_values = [optimizer.learning_rate.numpy()]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        for value in values:
            self.optimizer.learning_rate.assign(value)

    def _add_noise(self, lrs, t):
        if self._is_apply_noise(t):
            noise = self._calculate_noise(t)
            lrs = [v + v * noise for v in lrs]
        return lrs

    def _is_apply_noise(self, t) -> bool:
        """Return True if scheduler is in noise range."""
        apply_noise = False
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
        return apply_noise

    def _calculate_noise(self, t) -> float:
        np.random.seed(self.noise_seed + t)
        if self.noise_type == 'normal':
            while True:
                noise = np.random.randn()
                if abs(noise) < self.noise_pct:
                    return noise
        else:
            noise = 2 * (np.random.rand() - 0.5) * self.noise_pct
        return noise
