# Shows a progress bar for SB3 training

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

class Progress_Bar_Callback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")

    def _on_step(self) -> bool:
        self.pbar.update(self.model.n_envs)  # update by number of envs
        return True

    def _on_training_end(self):
        self.pbar.close()
