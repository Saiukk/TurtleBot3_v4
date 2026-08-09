from .log_utils import create_logger, init_wandb
import numpy as np
import time, os
import warnings; warnings.filterwarnings("ignore")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLoggingCallback(BaseCallback):
    """
    Mirrors PPO_SB3.EpisodeLoggingCallback's per-episode bookkeeping (reward/step/
    cost/collision/success averaged over the last `last_n` episodes), so SAC runs
    produce metrics comparable to DDQN's and PPO_SB3's. Only the diagnostics read
    out of SB3's internal logger differ, since SAC logs different training values.
    """

    # How often (in completed episodes) to print SAC's internal training diagnostics.
    METRICS_INTERVAL = 25

    # SAC.train() logs these via self.model.logger.record(...) once per gradient step
    # (every `train_freq` env steps -- every step, by default); read the latest values
    # straight out of that logger. "train/ent_coef_loss" is only recorded when ent_coef
    # is learned automatically (SB3's default, ent_coef="auto", used here).
    METRIC_KEYS = [
        "train/n_updates",
        "train/ent_coef",
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef_loss",
    ]

    def __init__(self, metrics_logger, wandb_log, last_n, model_dir, n_episode, run_name):
        super().__init__()
        # Named distinctly from `self.logger`: BaseCallback reserves that name for
        # SB3's own internal Logger (assigned via init_callback() when learn() starts),
        # which would silently clobber a plain custom logger stored under that name.
        self.metrics_logger = metrics_logger
        self.wandb_log = wandb_log
        self.last_n = last_n
        self.model_dir = model_dir
        self.n_episode = n_episode
        self.run_name = run_name

        self.reward_hist, self.cost_hist, self.step_hist, self.success_hist = [], [], [], []
        self.collision_hist = []
        self.ep_reward, self.ep_cost, self.ep_step = 0.0, 0, 0
        self.ep_collision = 0

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
        self.ep_collision += info["collision"]

        if done:
            episode = len(self.reward_hist)
            self.reward_hist.append(self.ep_reward)
            self.cost_hist.append(self.ep_cost)
            self.step_hist.append(self.ep_step)
            self.success_hist.append(1 if info["goal_reached"] else 0)
            self.collision_hist.append(self.ep_collision)

            last_n = min(len(self.reward_hist), self.last_n)
            reward_last_n = self.reward_hist[-last_n:]
            cost_last_n = self.cost_hist[-last_n:]
            step_last_n = self.step_hist[-last_n:]
            success_last_n = self.success_hist[-last_n:]
            collision_last_n = self.collision_hist[-last_n:]

            record = {
                'Episode': episode,
                'Step': int(np.mean(step_last_n)),
                'Avg_Cost': int(np.mean(cost_last_n) * 100),
                'Avg_Success': int(np.mean(success_last_n) * 100),
                'Avg_Reward': np.mean(reward_last_n),
                'Avg_Collision': int(np.mean(collision_last_n) * 100),
            }
            self.metrics_logger.write(record)
            if self.wandb_log:
                import wandb
                wandb.log(record)

            print(f"(SAC_SB3) Ep: {episode:5}", end=" ")
            print(f"reward: {self.ep_reward:5.2f} (last_{last_n}: {np.mean(reward_last_n):5.2f})", end=" ")
            print(f"cost_last_{last_n}: {int(np.mean(cost_last_n))}", end=" ")
            print(f"collision_last_{last_n}: {int(np.mean(collision_last_n) * 100)}%", end=" ")
            print(f"step_last_{last_n} {int(np.mean(step_last_n)):3d}", end=" ")
            print(f"success_last_{last_n} {int(np.mean(success_last_n) * 100):4d}%")

            if episode % self.METRICS_INTERVAL == 0:
                self._print_sac_diagnostics()

            # save model only the first time a new best avg_success (>= 79%) is reached,
            # not on every episode spent above the threshold
            current_success = int(np.mean(success_last_n) * 100)
            if current_success >= 79 and current_success > self.best_success:
                self.best_success = current_success
                os.makedirs(self.model_dir, exist_ok=True)
                # Overwrite in place: one file per run, always the current best,
                # instead of a new file for every improved success threshold.
                self.model.save(os.path.join(self.model_dir, self.run_name))

            self.ep_reward, self.ep_cost, self.ep_step = 0.0, 0, 0
            self.ep_collision = 0

            # Returning False stops model.learn() here: most episodes end well before
            # the 300-step timeout (goal/collision), so gating purely on total_timesteps
            # (n_episode * STEP_LIMIT) lets far more than n_episode episodes run before
            # that budget is used up. This makes --n_episode mean what it means for DDQN.
            if len(self.reward_hist) >= self.n_episode:
                return False

        return True

    def _print_sac_diagnostics(self):
        # self.logger here is SB3's internal Logger (see the note in __init__), the
        # same one SAC.train() writes to -- values are only present after the first
        # gradient step (train_freq env steps in), so skip silently until then.
        values = self.logger.name_to_value
        if "train/actor_loss" not in values:
            return

        line = (
            f"        [SAC] n_updates: {int(values['train/n_updates'])}  "
            f"ent_coef: {values['train/ent_coef']:.4f}  "
            f"actor_loss: {values['train/actor_loss']:.4f}  "
            f"critic_loss: {values['train/critic_loss']:.4f}"
        )
        if "train/ent_coef_loss" in values:
            line += f"  ent_coef_loss: {values['train/ent_coef_loss']:.4f}"
        print(line)


class SAC_SB3():

    """
    Thin wrapper around stable-baselines3's SAC, kept API-compatible with
    DDQN(env, args) / .loop(args) so training.py can select it interchangeably
    with DDQN/PPO_SB3.

    Requires a continuous action space (gym.spaces.Box): set
    CustomAgent.useContinuousActions=true on the agent in the Unity scene before
    training with this algorithm. SAC asserts on the Discrete(3) action space
    DDQN/PPO_SB3 normally use, since it's a deterministic-entropy continuous-control
    algorithm with no discrete-action variant.
    """

    # Matches RoboticNavigation's internal step-limit timeout (env/robotic_navigation.py)
    STEP_LIMIT = 300

    # Env steps of pure random-action exploration collected before the first gradient
    LEARNING_STARTS = 2000

    def __init__(self, env, args):

        self.env = env
        self.run_name = f"{args.alg}__{args.tag if args.tag != '' else ''}__{args.seed}__{int(time.time())}"

        self.model = SAC(
            "MlpPolicy",
            env,
            gamma=args.gamma,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            tau=args.tau,
            gradient_steps=args.n_epochs,
            learning_starts=self.LEARNING_STARTS,
            seed=args.seed,
            verbose=0,
        )

    def loop(self, args):

        # Initialize the logger
        metrics_logger = create_logger(self.run_name, args)
        if args.wandb_log: init_wandb(self.run_name, args)

        callback = EpisodeLoggingCallback(
            metrics_logger, args.wandb_log, last_n=args.last_n, model_dir="models", n_episode=args.n_episode,
            run_name=self.run_name
        )

        # The callback stops training once args.n_episode episodes complete (matching
        # DDQN's semantics). total_timesteps is only a safety ceiling in case episodes
        # ran the full length every time; it's never expected to be the binding limit.
        total_timesteps = args.n_episode * self.STEP_LIMIT
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
