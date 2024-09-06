import torch as th
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import numpy as np
from stable_baselines3.common.type_aliases import RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv


def learn(
    policy,
    total_timesteps: int,
    callback = None,
    log_interval: int = 4,
    tb_log_name: str = "run",
    reset_num_timesteps: bool = True,
    progress_bar: bool = False,
):
    total_timesteps, callback = policy._setup_learn(
        total_timesteps,
        callback,
        reset_num_timesteps,
        tb_log_name,
        progress_bar,
    )

    callback.on_training_start(locals(), globals())

    assert policy.env is not None, "You must set the environment before calling learn()"

    while policy.num_timesteps < total_timesteps:
        rollout = collect_rollouts(
            policy,
            policy.env,
            train_freq=policy.train_freq,
            action_noise=policy.action_noise,
            callback=callback,
            learning_starts=policy.learning_starts,
            replay_buffer=policy.replay_buffer,
            log_interval=log_interval,
        )

        if not rollout.continue_training:
            break

        if policy.num_timesteps > 0 and policy.num_timesteps > policy.learning_starts:
            # If no `gradient_steps` is specified,
            # do as many gradients steps as steps performed during the rollout
            gradient_steps = policy.gradient_steps if policy.gradient_steps >= 0 else rollout.episode_timesteps
            # Special case when the user passes `gradient_steps=0`
            if gradient_steps > 0:
                policy.train(batch_size=policy.batch_size, gradient_steps=gradient_steps)

    callback.on_training_end()

    return policy
    
def collect_rollouts(
    policy,
    env,
    callback,
    train_freq,
    replay_buffer,
    action_noise = None,
    learning_starts: int = 0,
    log_interval= None,
):
    """
    Collect experiences and store them into a ``ReplayBuffer``.

    :param env: The training environment
    :param callback: Callback that will be called at each step
        (and at the beginning and end of the rollout)
    :param train_freq: How much experience to collect
        by doing rollouts of current policy.
        Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
        or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
        with ``<n>`` being an integer greater than 0.
    :param action_noise: Action noise that will be used for exploration
        Required for deterministic policy (e.g. TD3). This can also be used
        in addition to the stochastic policy for SAC.
    :param learning_starts: Number of steps before learning for the warm-up phase.
    :param replay_buffer:
    :param log_interval: Log data every ``log_interval`` episodes
    :return:
    """
    # Switch to eval mode (this affects batch norm / dropout)
    policy.policy.set_training_mode(False)

    num_collected_steps, num_collected_episodes = 0, 0

    
    if env.num_envs > 1:
        assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

    if policy.use_sde:
        policy.actor.reset_noise(env.num_envs)

    callback.on_rollout_start()
    continue_training = True
    while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        if policy.use_sde and policy.sde_sample_freq > 0 and num_collected_steps % policy.sde_sample_freq == 0:
            # Sample a new noise matrix
            policy.actor.reset_noise(env.num_envs)

        # Select action randomly or according to policy
        actions, buffer_actions = policy._sample_action(learning_starts, action_noise, env.num_envs)

        # Rescale and perform action
        new_obs, rewards, dones, infos = env.step(actions)

        policy.num_timesteps += env.num_envs
        num_collected_steps += 1

        # Give access to local variables
        callback.update_locals(locals())
        # Only stop training if return value is False, not when it is None.
        if not callback.on_step():
            return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

        # Retrieve reward and episode length if using Monitor wrapper
        policy._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        policy._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

        policy._update_current_progress_remaining(policy.num_timesteps, policy._total_timesteps)

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        policy._on_step()

        for idx, done in enumerate(dones):
            if done:
                # Update stats
                num_collected_episodes += 1
                policy._episode_num += 1

                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)

                # Log training infos
                if log_interval is not None and policy._episode_num % log_interval == 0:
                    policy._dump_logs()
    callback.on_rollout_end()

    return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)
