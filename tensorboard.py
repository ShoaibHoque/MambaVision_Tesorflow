import tensorflow as tf

class TensorboardLogger:
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        with self.writer.as_default():
            for k, v in kwargs.items():
                if v is None:
                    continue
                if isinstance(v, tf.Tensor):
                    v = v.numpy()
                assert isinstance(v, (float, int))
                tf.summary.scalar(head + "/" + k, v, step=self.step if step is None else step)

    def flush(self):
        self.writer.flush()
