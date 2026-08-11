import gym;

gym.logger.set_level(40)
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import os, sys, time


class RoboticNavigation(gym.Env):
    """
	A class that implements a wrapper between the Unity Engine environment and a custom Gym environment.

	This version wraps the TWO-agent scene. Unity exposes one behavior per agent
	("agent_navigation" for Agent, "agent_navigation_2" for Agent2; each behavior
	holds exactly one agent that requests a decision every step). The wrapper
	aggregates the two agents into a single joint Gym environment so the existing
	single-agent algorithms (DDQN / PPO / SAC) keep working unchanged:

		observation : [agent0, agent1]          -> 15 + 15 = 30 dims, in [0, 1]
		              each agent = lidar(11) + target angle + target distance
		                           + other-agent angle + other-agent distance
		action      : [lin0, ang0, lin1, ang1]  -> Box(4), linear in [0, 1] and
		              angular in [-1, 1] (no reverse, see CustomAgent.cs)
		reward      : sum of the two agents' (shaped) rewards, plus a team-success
		              bonus (+2) when both agents reach their targets; crashes are
		              penalized at -3 for the crashing agent only
		done        : True only when BOTH agents have reached their own goals
		              (a finisher idles frozen on its target meanwhile), when
		              either agent crashes, or at the step limit
		info        : aggregate flags (goal_reached, collision, cost, time_out)
		              plus a per-agent breakdown

	The main motivations for this wrapper are:

		1) Fix the state
			originally the LiDAR scans arrive with a size of 2 * number_of_scan, because for each direction Unity returns two values, the
			first one is a float that represent the distance from the first obstacle, normalized between [0, 1]. The second one is a flag integer [0, 1]
			which indicates if there is an obstacle in the range of the corresponding scan. To avoid a strong correlation between the sensors input of the network,
			we removed the flag value. This is also to increase the explainability of the state value (useful also for the properties).

		2) Change the reward
			this wrapper allows us to change the reward function without modifying the Unity3D project.
	"""

    def __init__(self, step_limit=300, worker_id=0, editor_build=False, env_type="training", random_seed=0,
                 render=False, no_graphics=True):

        """
		Constructor of the class.

		Parameters
		----------
			rendered : bool
				flag to run the environment in rendered mode, currently unused (default: False)
			no_graphics : bool
				run a standalone build headless (-nographics -batchmode); no-op for the Editor
		"""

        # Load the scan number given in input. Must equal 2 * RaysPerDirection + 1,
        # matching the Ray Perception Sensor Component 3D config in the Unity scene
        # (5 rays per direction -> 11 total rays, vs. the previous 3 -> 7).
        self.scan_number = 11
        self.step_limit = step_limit

        # If the env_path is given as input override the environment search
        if not editor_build:

            # Sanity check for the 'env_type' option
            assert_message = "Invalid env_type, options [None, training, render]"
            assert (env_type in [None, "training", "render", "gym", "testing"]), assert_message

            # Detect the current platform and load the corresponding environment
            # path for the selected 'env_type' (training/render/gym/testing).
            if sys.platform.startswith("linux"):
                platform_dir = "linux"
            elif sys.platform == "darwin":
                platform_dir = "macos"
            elif os.name == "nt":
                platform_dir = "windows"
            else:
                raise NotImplementedError(f"Unsupported platform: {sys.platform}")

            env_path = f"env/{platform_dir}_{env_type}/SafeRobotics"

        # For the editor build force the path to None and the worker id to 0,
        # assigned values for the editor build.
        else:
            env_path = None
            worker_id = 0

        # Load the Unity Environment directly (NOT through gym_unity's
        # UnityToGymWrapper: that wrapper asserts a single behavior and a single
        # agent, which the two-agent scene violates). Uses the mlagents_envs
        # 0.28 API: reset()/step() signal the sim, and get_steps(behavior)
        # returns (DecisionSteps, TerminalSteps) per behavior.
        self.unity_env = UnityEnvironment(env_path, worker_id=worker_id, seed=random_seed,
                                          no_graphics=no_graphics)

        # The behavior list is only populated once the initial handshake with
        # Unity completes; force it with a no-op step when it is still empty.
        if not self.unity_env.behavior_specs:
            self.unity_env.step()

        # Deterministic agent order: "agent_navigation" (Agent) first, then
        # "agent_navigation_2" (Agent2). Keeps the joint state/action layout
        # stable across runs.
        self.behavior_names = sorted(self.unity_env.behavior_specs.keys())
        self.n_agents = len(self.behavior_names)
        assert self.n_agents >= 1, "No behaviors found in the Unity environment"

        # Per-agent raw vector observation size: lidar (2 * scan_number) + target
        # angle + target distance + other-agent angle + other-agent distance + cost.
        obs_specs = self.unity_env.behavior_specs[self.behavior_names[0]].observation_specs
        self.raw_vec_size = sum(s.shape[0] for s in obs_specs if len(s.shape) == 1)

        # The scene's CustomAgent.cs overrides the behavior to continuous(2) at
        # runtime when useContinuousActions is enabled, so the spec reported to
        # Python here is continuous.
        action_spec = self.unity_env.behavior_specs[self.behavior_names[0]].action_spec
        assert action_spec.is_continuous(), (
            "The two-agent wrapper currently requires continuous actions "
            "(set CustomAgent.useContinuousActions = true on both agents in the scene)"
        )
        self.action_size_per_agent = action_spec.continuous_size  # 2: linear, angular

        # Fixed state layout per agent: lidar(scan_number) + target angle + target
        # distance + other-agent angle + other-agent distance. The trailing cost
        # flag is dropped, exactly like the single-agent wrapper did.
        self.tail_size = self.raw_vec_size - 2 * self.scan_number - 1
        self.state_size = self.scan_number + self.tail_size
        # Index of the target distance within the fixed state (reward shaping input).
        self.distance_idx = self.scan_number + 1

        # Joint action space: [lin0, ang0, lin1, ang1, ...]. Linear only drives
        # forward (CustomAgent.cs clamps it to [0, 1]: the LiDAR can't see behind
        # the robot), angular uses the full [-1, 1] range so each agent can blend
        # a turn with forward motion into an arc.
        self.action_space = gym.spaces.Box(
            low=np.array([0, -1] * self.n_agents, dtype=np.float32),
            high=np.array([1, 1] * self.n_agents, dtype=np.float32),
            dtype=np.float32,
        )

        # Joint observation space, every component normalized in [0, 1]
        # (lidar distances, target angle, target distance, other-agent angle and
        # distance are all normalized in Unity's CustomAgent.cs).
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.state_size * self.n_agents, dtype=np.float32),
            high=np.ones(self.state_size * self.n_agents, dtype=np.float32),
            dtype=np.float32,
        )

        # Per-agent bookkeeping
        self.step_counter = 0
        self.target_distance = [0.0] * self.n_agents
        # Latch of which agents have already reached their own target. An agent
        # that finishes is frozen in Unity and idles while the episode keeps
        # running for the others; it is reset together with everyone else.
        self.finished = [False] * self.n_agents

    # ---------------------------------------------------------------------------
    # Gym interface
    # ---------------------------------------------------------------------------

    def reset(self):

        """
		Override of the reset function of OpenAI Gym

		Returns
		----------
			state : np.ndarray
				the concatenated observation of the two agents, with
				state_size * n_agents elements, each agent's portion being
				[scan_number lidar values, target angle, target distance,
				other-agent angle, other-agent distance], all normalized in [0, 1]
		"""

        # Reset the counter for the maximum step counter
        self.step_counter = 0

        # No agent has reached its target at the start of a fresh episode
        self.finished = [False] * self.n_agents

        # Signal Unity to reset the simulation (calls OnEpisodeBegin on every agent)
        self.unity_env.reset()

        # Gather the initial observations of all the agents
        gathered = self._gather()

        # Store the distance from the target of each agent for the reward function
        self.target_distance = [state[self.distance_idx] for state, _, _ in gathered]

        # Concatenate the per-agent states into the joint observation
        return self._joint_state(gathered)

    def step(self, action):

        """
		Override of the step function of OpenAI Gym

		Parameters
		----------
			action : np.ndarray
				joint action [lin0, ang0, lin1, ang1] (Box(4) for two agents)

		Returns
		----------
			state : np.ndarray
				the concatenated observation of the two agents (see reset)
			reward : float
				sum of the (shaped) rewards of the two agents
			done : bool
				flag that indicates if the current episode is terminal
			info : dict
				aggregate flags (goal_reached, collision, cost, time_out) plus a
				per-agent breakdown
		"""

        # Split the joint action into per-agent actions and send them to Unity.
        # Each behavior owns a single agent, so the action has shape (1, 2).
        joint_action = np.asarray(action, dtype=np.float32).reshape(self.n_agents, -1)
        for i, name in enumerate(self.behavior_names):
            action_tuple = ActionTuple()
            action_tuple.add_continuous(joint_action[i:i + 1])
            self.unity_env.set_actions(name, action_tuple)

        # Advance the simulation by one step
        self.unity_env.step()
        self.step_counter += 1

        # Gather the new observations and rewards of all the agents
        gathered = self._gather()
        states = [g[0] for g in gathered]
        raw_states = [g[1] for g in gathered]
        rewards = [g[2] for g in gathered]

        # Remember which agents were already finished before this step: a finished
        # agent keeps reporting reward 1 while frozen on its target, and that must
        # not be re-counted as a fresh arrival nor re-rewarded.
        was_finished = list(self.finished)

        # Per-agent terminal signals from Unity: reward 1 means the agent reached
        # ITS OWN target (CustomAgent.cs only rewards the owner's target now), -1
        # means a crash. The finished flag is latched: it stays set while the
        # agent idles frozen on its target, even though reward keeps reading 1.
        for i, r in enumerate(rewards):
            if r == 1 and not was_finished[i]:
                self.finished[i] = True

        collision = any(r == -1 for r in rewards)
        time_out = (self.step_counter >= self.step_limit)

        # Aggregate info from the single agents (same keys as the old single-agent
        # wrapper, so the algorithms' metric callbacks keep working unchanged)
        info = {
            # Team success: every agent reached its own target. This is the flag the
            # algorithms log as the success metric.
            "goal_reached": all(self.finished),
            "collision": collision,
            "cost": any(raw[-1] == 1 for raw in raw_states),
            "time_out": time_out,
            # Per-agent breakdown, handy for debugging / per-agent metrics
            "per_agent": {
                name: {
                    "reward": r,
                    "goal_reached": r == 1,
                    "collision": r == -1,
                    "cost": raw[-1] == 1,
                }
                for name, (_, raw, r) in zip(self.behavior_names, gathered)
            },
        }

        # Team-done semantics: the episode ends only when BOTH agents have reached
        # their own goals (a finisher idles frozen on its target meanwhile), when
        # either agent crashes, or at the step limit. This matches the standard
        # cooperative-MARL setup and lets the slower agent train the full episode
        # instead of being truncated every time the faster one finishes.
        done = (info["goal_reached"] or collision or time_out)

        # Shape the reward per agent (mirrors the single-agent wrapper), then sum.
        # Already-finished agents contribute 0: no more shaping and no per-step
        # penalty while they idle -- the waiting time is the other agent's cost.
        shaped = [
            self.override_reward(state, reward, i, done, was_finished[i])
            for i, (state, reward) in enumerate(zip(states, rewards))
        ]
        joint_reward = float(np.sum(shaped))

        # Team-success bonus: on the step where the LAST agent reaches its target
        # (info["goal_reached"] == all agents finished), add the shared bonus once.
        if info["goal_reached"]:
            joint_reward += self.success_bonus

        # Concatenate the per-agent states into the joint observation
        state = self._joint_state(gathered)

        return state, joint_reward, done, info

    def _joint_state(self, gathered):
        """Concatenate the fixed states of all agents into the joint observation."""
        return np.concatenate([s for s, _, _ in gathered]).astype(np.float32)

    def _gather(self):
        """
		Collect the latest observation, raw state and reward of every agent.

		Returns
		----------
			list of (fixed_state, raw_state, reward) tuples, one per behavior.
			The reward is the one accumulated by that agent since the last step
			(DecisionSteps.reward) or, if the agent terminated in Unity, the one
			reported by the TerminalSteps (in this project agents never call
			EndEpisode: episode termination is handled here in Python).
		"""
        out = []
        for name in self.behavior_names:
            decision_steps, terminal_steps = self.unity_env.get_steps(name)
            if len(terminal_steps) > 0:
                steps = terminal_steps
            else:
                steps = decision_steps
            assert len(steps) > 0, (
                f"Behavior '{name}' reported no agent ready for a decision; "
                "expected exactly one agent per behavior"
            )

            # Concatenate all the vector observations of the agent (lidar + the
            # CustomAgent extra observations) into a single raw state.
            vec_obs = [o for o in steps.obs if o.ndim == 2]
            raw_state = np.concatenate(vec_obs, axis=1)[0].astype(np.float32)
            reward = float(steps.reward[0])

            fixed_state = self.fix_state(raw_state)
            assert len(fixed_state) == self.state_size, (
                f"Unexpected fixed state size {len(fixed_state)} != {self.state_size}; "
                "check scan_number / lidar config against the scene"
            )
            out.append((fixed_state, raw_state, reward))
        return out

    # Collisions are scaled up relative to Unity's raw -1: the per-step penalty below
    # (needed to stop the agent from stalling/rotating in place) also rewards shaving
    # steps off a route, so cutting a corner to save a few steps can pay off even at
    # some collision risk unless the collision penalty clearly dominates that saving.
    # Scaled by 3 (was 5): with the team-success bonus added below, success now
    # dominates crash by a clear margin while a lower scale keeps the policy from
    # becoming over-conservative (parking to avoid any collision risk).
    collision_penalty_scale = 3

    # Shared team reward granted on the exact step the LAST agent reaches its
    # target. It widens the gap between team success (both goals + arrivals + bonus)
    # and crash/timeout returns, so the critics clearly separate "both agents made
    # it" from everything else -- this is what most drives the coordination.
    success_bonus = 2.0

    def override_reward(self, state, reward, agent_idx, done, was_finished=False):

        # The exact step the agent first reaches its own target: +1 (Unity's
        # SetReward). Reward stays 1 while the agent idles frozen on the target
        # afterwards, so the was_finished latch is what keeps the +1 from being
        # granted again on every step of the wait.
        if reward == 1 and not was_finished:
            return reward

        # Crash: amplify the -1 so collision clearly dominates the per-step savings
        # that cutting a corner can earn (see collision_penalty_scale comment).
        # done is always True on a crash; the guard keeps a hypothetical non-terminal
        # -1 from being mis-scaled.
        if reward == -1:
            return reward * (self.collision_penalty_scale if done else 1)

        # Already-finished agent idling on its target: contributes nothing. It no
        # longer moves, so shaping and the per-step penalty don't apply -- waiting
        # for the other agent costs the team nothing extra.
        if was_finished:
            return 0

        # Other terminal states (timeout) already carry the correct value (0 from
        # Unity): don't shape those, just pass them through.
        if done:
            return reward

        # state[self.distance_idx] is the geodesic (NavMesh) distance to this agent's
        # own target, not the straight-line one, so a corridor detour that's genuinely
        # on the shortest path decreases it like any other progress instead of being
        # penalized.
        reward_multiplier, step_penalty = 3, 0.01
        new_distance = state[self.distance_idx]
        distance_delta = self.target_distance[agent_idx] - new_distance
        self.target_distance[agent_idx] = new_distance

        # Clip: NavMesh corner-routing can shift discretely step to step (unlike the
        # smooth straight-line distance), so cap the shaping term to avoid single-step
        # reward spikes. step_penalty is large enough relative to this cap that idling
        # for a full episode (step_penalty * step_limit) costs more than the terminal
        # reward, removing the incentive to rotate in place instead of committing.
        return np.clip(distance_delta * reward_multiplier, -0.05, 0.05) - step_penalty

    def fix_state(self, state):

        """
		Support function to convert the observation vector from the version obtained by Unity3D to our configuration.
		The original version provides a 2*scan_number size for the LiDAR,
		for each direction 2 value, one with the float value and one with the flag [0, 1].
		In this wrapper we remove the flag, maintaining only one value for each direction

		Parameters
		----------
			state : np.ndarray
				the raw observation of a single agent from the environment

		Returns
		----------
			state : np.ndarray
				an observation of scan_number + tail_size elements: the lidar scan
				(ordered as the network expects), the target angle and distance and
				the other-agent angle and distance. All values normalized in [0, 1].
		"""

        # Compute the size of the observation array that correspond to the lidar sensor,
        # the other portion is maintained
        scan_limit = 2 * (self.scan_number)
        state_lidar = [s for id, s in enumerate(state[:scan_limit]) if id % 2 == 1]

        # Change the order of the lidar scan to the order of the wrapper (see the class declaration for details)
        lidar_ordered_1 = [s for id, s in enumerate(reversed(state_lidar)) if id % 2 == 0]
        lidar_ordered_2 = [s for id, s in enumerate(state_lidar) if id % 2 == 1]
        lidar_ordered = lidar_ordered_1 + lidar_ordered_2

        # Concatenate the ordered lidar state with the other values of the state
        # (dropping the trailing cost flag, kept in the raw state for info["cost"])
        state_fixed = lidar_ordered + list(state[scan_limit:-1])

        #
        return np.array(state_fixed)

    # Override the "close" function
    def close(self):
        self.unity_env.close()

    # Override the "render" function
    def render(self):
        pass
