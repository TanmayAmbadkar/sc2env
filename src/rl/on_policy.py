import torch as th
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import numpy as np
from gymnasium import spaces
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

def learn(
        policy,
        total_timesteps: int,
        callback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = policy._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert policy.env is not None

        while policy.num_timesteps < total_timesteps:
            continue_training = collect_rollouts(policy, policy.env, callback, policy.rollout_buffer, n_rollout_steps=policy.n_steps, use_masking = use_masking)

            if not continue_training:
                break

            iteration += 1
            policy._update_current_progress_remaining(policy.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert policy.ep_info_buffer is not None
                policy._dump_logs(iteration)

            policy.train()

        callback.on_training_end()

        return policy
    
def collect_rollouts(
        policy,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert policy._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        policy.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if policy.use_sde:
            policy.policy.reset_noise(env.num_envs)
                    
        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper")


        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if policy.use_sde and policy.sde_sample_freq > 0 and n_steps % policy.sde_sample_freq == 0:
                # Sample a new noise matrix
                policy.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(policy._last_obs, policy.device)
                
                if use_masking:
                    action_masks = get_action_masks(env)

                    actions, values, log_probs = policy.policy(obs_tensor, action_masks=action_masks)
                
                actions, values, log_probs = policy.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(policy.action_space, spaces.Box):
                if policy.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = policy.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, policy.action_space.low, policy.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            policy.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            policy._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(policy.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = policy.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = policy.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += policy.gamma * terminal_value

            if use_masking:
                rollout_buffer.add(
                    policy._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    policy._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                    action_masks = action_masks
                )
            else:
                rollout_buffer.add(
                    policy._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    policy._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                )
            policy._last_obs = new_obs  # type: ignore[assignment]
            policy._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = policy.policy.predict_values(obs_as_tensor(new_obs, policy.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
