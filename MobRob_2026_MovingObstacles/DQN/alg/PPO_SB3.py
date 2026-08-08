from .log_utils import create_logger, init_wandb
import numpy as np
import time, os
import warnings; warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLoggingCallback(BaseCallback):
    """
    Mirrors DDQN.loop's per-episode bookkeeping (reward/step/cost/success averaged
    over the last `last_n` episodes) so PPO runs produce metrics comparable to DDQN's.
    """

    # How often (in completed episodes) to print PPO's internal training diagnostics.
    METRICS_INTERVAL = 25

    # PPO.train() logs these via self.model.logger.record(...) once per policy update
    # (every N_STEPS env steps); read the latest values straight out of that logger.
    METRIC_KEYS = [
        "train/explained_variance",
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/n_updates",
    ]

    def __init__(self, metrics_logger, wandb_log, last_n, model_dir, n_episode):
        super().__init__()
        # Named distinctly from `self.logger`: BaseCallback reserves that name for
        # SB3's own internal Logger (assigned via init_callback() when learn() starts),
        # which would silently clobber a plain custom logger stored under that name.
        self.metrics_logger = metrics_logger
        self.wandb_log = wandb_log
        self.last_n = last_n
        self.model_dir = model_dir
        self.n_episode = n_episode

        self.reward_hist, self.cost_hist, self.step_hist, self.success_hist = [], [], [], []
        self.ep_reward, self.ep_cost, self.ep_step = 0.0, 0, 0

        # Tracks the best rolling success seen so far, so only genuine improvements
        # get saved instead of one file per episode for the entire stretch spent
        # above the save threshold.
        self.best_success = -1

    def _on_step(self):

        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]

        self.ep_reward += reward
        self.ep_step += 1
        self.ep_cost += info["cost"]

        if done:
            episode = len(self.reward_hist)
            self.reward_hist.append(self.ep_reward)
            self.cost_hist.append(self.ep_cost)
            self.step_hist.append(self.ep_step)
            self.success_hist.append(1 if info["goal_reached"] else 0)

            last_n = min(len(self.reward_hist), self.last_n)
            reward_last_n = self.reward_hist[-last_n:]
            cost_last_n = self.cost_hist[-last_n:]
            step_last_n = self.step_hist[-last_n:]
            success_last_n = self.success_hist[-last_n:]

            record = {
                'Episode': episode,
                'Step': int(np.mean(step_last_n)),
                'Avg_Cost': int(np.mean(cost_last_n) * 100),
                'Avg_Success': int(np.mean(success_last_n) * 100),
                'Avg_Reward': np.mean(reward_last_n),
            }
            self.metrics_logger.write(record)
            if self.wandb_log:
                import wandb
                wandb.log(record)

            print(f"(PPO_SB3) Ep: {episode:5}", end=" ")
            print(f"reward: {self.ep_reward:5.2f} (last_{last_n}: {np.mean(reward_last_n):5.2f})", end=" ")
            print(f"cost_last_{last_n}: {int(np.mean(cost_last_n))}", end=" ")
            print(f"step_last_{last_n} {int(np.mean(step_last_n)):3d}", end=" ")
            print(f"success_last_{last_n} {int(np.mean(success_last_n) * 100):4d}%")

            if episode % self.METRICS_INTERVAL == 0:
                self._print_ppo_diagnostics()

            # save model only the first time a new best avg_success (>= 79%) is reached,
            # not on every episode spent above the threshold
            current_success = int(np.mean(success_last_n) * 100)
            if current_success >= 79 and current_success > self.best_success:
                self.best_success = current_success
                os.makedirs(self.model_dir, exist_ok=True)
                self.model.save(os.path.join(self.model_dir, f"PPO_SB3_ep{episode}_success{current_success}"))

            self.ep_reward, self.ep_cost, self.ep_step = 0.0, 0, 0

            # Returning False stops model.learn() here: most episodes end well before
            # the 300-step timeout (goal/collision), so gating purely on total_timesteps
            # (n_episode * STEP_LIMIT) lets far more than n_episode episodes run before
            # that budget is used up. This makes --n_episode mean what it means for DDQN.
            if len(self.reward_hist) >= self.n_episode:
                return False

        return True

    def _print_ppo_diagnostics(self):
        # self.logger here is SB3's internal Logger (see the note in __init__), the
        # same one PPO.train() writes to -- values are only present after the first
        # policy update (every N_STEPS env steps), so skip silently until then.
        values = self.logger.name_to_value
        if "train/explained_variance" not in values:
            return

        print(
            f"        [PPO] explained_variance: {values['train/explained_variance']:6.3f}  "
            f"approx_kl: {values['train/approx_kl']:.4f}  "
            f"clip_fraction: {values['train/clip_fraction']:.3f}  "
            f"entropy_loss: {values['train/entropy_loss']:.4f}  "
            f"value_loss: {values['train/value_loss']:.4f}  "
            f"policy_grad_loss: {values['train/policy_gradient_loss']:.4f}  "
            f"n_updates: {int(values['train/n_updates'])}"
        )


class PPO_SB3():

    """
    Thin wrapper around stable-baselines3's PPO, kept API-compatible with
    DDQN(env, args) / .loop(args) so training.py can select either interchangeably.
    Uses the default MlpPolicy, which auto-detects the env's Discrete(3) action space
    (unlike alg/PPO_PT.py, a custom implementation built for continuous control).
    """

    # Matches RoboticNavigation's internal step-limit timeout (env/robotic_navigation.py)
    STEP_LIMIT = 300

    # PPO-specific hyperparameters not exposed via config.py, hardcoded here:
    N_STEPS = 2048       
    GAE_LAMBDA = 0.95   
    CLIP_RANGE = 0.2    
    VF_COEF = 0.5       
    ENT_COEF = 0.01    

    def __init__(self, env, args):

        self.env = env
        self.run_name = f"{args.alg}__{args.tag if args.tag != '' else ''}__{args.seed}__{int(time.time())}"

        self.model = PPO(
            "MlpPolicy",
            env,
            gamma=args.gamma,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            seed=args.seed,
            n_steps=self.N_STEPS,
            gae_lambda=self.GAE_LAMBDA,
            clip_range=self.CLIP_RANGE,
            vf_coef=self.VF_COEF,
            ent_coef=self.ENT_COEF,
            verbose=0,
        )

    def loop(self, args):

        # Initialize the logger
        metrics_logger = create_logger(self.run_name, args)
        if args.wandb_log: init_wandb(self.run_name, args)

        callback = EpisodeLoggingCallback(
            metrics_logger, args.wandb_log, last_n=args.last_n, model_dir="models", n_episode=args.n_episode
        )

        # The callback stops training once args.n_episode episodes complete (matching
        # DDQN's semantics). total_timesteps is only a safety ceiling in case episodes
        # ran the full length every time; it's never expected to be the binding limit.
        total_timesteps = args.n_episode * self.STEP_LIMIT
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
