import numpy as np
import tensorflow as tf
from Agents.PPO.Memory import Memory
# from Agents.PPO.Networks import ActorLstm, CriticLstm
from Agents.PPO.Networks import Network



import tensorflow_probability as tfp # type: ignore


class Agent():
    def __init__(self, pool, **kwargs):
        
        self.pool = pool

        app_config = kwargs["app_config"]
        
        # initial hyper params
        self.n_batches =        app_config.n_batches
        self.batch_inc_freq =   app_config.batch_inc_freq
        self.lstm_params =      app_config.lstm_params
        self.alpha =            app_config.alpha
        self.k_epochs =         app_config.k_epochs
        self.entropy_coeff =    app_config.entropy_coeff

        self.graph_path_dir =   app_config.graph_path_dir

        self.clip_pram = 0.2
        self.epoch_count = 0



        # Models, Optimizers initialization
        print(f"Initializing temp env to collect env params")
        temp_env = self.pool.get_random_env()
        self.sample_shape = temp_env.observation_space.shape
        self.signal_size = temp_env.action_space.shape[0]
        self.pool.close()

        # memories
        self.memory = Memory(state_shape=self.sample_shape)
        self.internal_state_index = 0
        self.channel_weights = np.zeros(self.signal_size, dtype=np.int32)
        self.returns_weights = np.zeros(self.signal_size, dtype=np.float32)
        self.channel_scores = np.zeros(self.signal_size, dtype=np.int32)

        # Initialize networks
        network = Network(self.sample_shape)
        self.policy_dict = network.get_policy(
            lstm_params=self.lstm_params, 
            signal_size=self.signal_size, 
            alpha=self.alpha
        )
        self.value_dict = network.get_value(
            lstm_params=self.lstm_params, 
            signal_size=self.signal_size, 
            alpha=self.alpha
        )
        

        

        # Log Graph tensorboard stuff
        # dir2 = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir2 = "lstm_" + "_".join(map(str, self.lstm_params))
        dir2 += f"alpha_{str(self.alpha)}"
        dir2 += f"k epoch_{str(self.k_epochs)}"
        dir2 += f"entropy_coeff_{str(self.entropy_coeff)}"
        

        
        
        self.summary_writer0 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/critic")
        self.summary_writer1 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/actor")
        self.summary_writer2 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/trend_selected")
        self.summary_writer3 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/trend_actual")
        self.summary_writer5 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/sigma")
        self.summary_writer6 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/reward")
        self.summary_writer7 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/batchsize")
        self.summary_writer8 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/max_trained_ts")


        self.sum_writer0 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/trained_ts0")
        self.sum_writer1 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/trained_ts1")
        self.sum_writer2 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/trained_ts2")
        self.sum_writer3 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/trained_ts3")
        self.sum_writer4 = tf.summary.create_file_writer(f"{self.graph_path_dir}/{dir2}_2/trained_ts4")
        



        # # Create a figure and axes



        
        
        



    @tf.function
    def choose_action(self, input1: tf.Tensor, input2: tf.Tensor, policy_states, value_states):
        """Selects an action by sampling from a policy combining discrete and continuous distributions.

        Args:
            inputs: Input tensor with shape [1, ...].
            policy_states: States for the policy model (e.g., RNN states).
            value_states: States for the value model.

        Returns:
            Dict containing the sampled action, index, log probability, value, and model states.
        """


        # 1. Continuous signal prediction
        # inject


        # model's call
        mu, sigma, policy_states = self.policy_dict['model'](
            inputs=(input1, input2),
            states=policy_states,
            training=False
        )

        values, value_states = self.value_dict['model'](
            inputs=(input1, input2), 
            states=value_states, 
            training=False
        )

        # 2. Check for valid model outputs
        tf.debugging.check_numerics(mu, "Policy mu contains NaN or inf")
        tf.debugging.check_numerics(sigma, "Policy sigma contains NaN or inf")
        tf.debugging.check_numerics(values, "Value contains NaN or inf")




        # 3. Make sample signals, make logprobs for each one
        dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        trend_samples = dist.sample() # (batchsize, 1)
        logprobs = dist.log_prob(trend_samples) # (batchsize, 1)


        
        tf.debugging.assert_shapes([
            (trend_samples, (1, 1)), # can be either 1d or 2d any logic
            (logprobs, (1, 1)), # can be either 1d or 2d any logic
            (mu, (1, 1)),
            (sigma, (1, 1)),
            (values, (1, 1))
        ])





        # 3. == All scalar values
        sample = tf.squeeze(trend_samples)
        logprob = tf.squeeze(logprobs)
        value = tf.squeeze(values)
        


        

        approach = {
            "sample": sample,
            "logprob": logprob,
            "value": value,
            "mu": tf.squeeze(mu),
            "sigma": tf.squeeze(sigma),
            "policy_states": policy_states,
            "value_states": value_states
        }

        return approach

    
            
    
    
    def train_step(
            self, 
            states: np.ndarray, 
            trend_samples: np.ndarray, 
            sample_indices: np.ndarray, 
            logprobs: np.ndarray,
            advantages: np.ndarray, 
            returns: np.ndarray
    ) -> tuple[tf.Tensor, tf.Tensor]:
        
        # === 1) BASIC TYPE & EMPTY CHECKS ===
        for name, item in [
            ("states", states),
            ("trend_samples", trend_samples),
            ("sample_indices", sample_indices),
            ("logprobs", logprobs),
            ("advantages", advantages),
            ("returns", returns),
        ]:
            if not isinstance(item, np.ndarray):
                raise ValueError(f"Expected `{name}` to be a NumPy array, got {type(item)}")
            if item.size == 0:
                raise ValueError(f"`{name}` is empty")
            if np.isnan(item).any() or np.isinf(item).any():
                raise ValueError(f"Invalid data: `{name}` contains NaN or Inf")

        # === 2) VALUE RANGE CHECKS === 
        # Add appropriate value range validations here if needed
        # For example:
        if np.any(sample_indices < 0) or np.any(sample_indices >= self.signal_size):
            raise ValueError(f"sample_indices must be in range [0, {self.signal_size-1}]")





        # === 3) SHAPE VALIDATION ===
        batch = returns.shape[0]
        expected_state_shape = (batch,) + self.sample_shape
        if states.shape != expected_state_shape:
            raise ValueError(
                f"`states` has shape {states.shape}, expected {expected_state_shape}"
            )
        if trend_samples.shape != (batch,):
            raise ValueError(f"`trend_samples` has shape {trend_samples.shape}, expected {(batch,)}")
        if sample_indices.shape != (batch,):
            raise ValueError(f"`sample_indices` has shape {sample_indices.shape}, expected {(batch,)}")
        if logprobs.shape != (batch,):
            raise ValueError(f"`logprobs` has shape {logprobs.shape}, expected {(batch,)}")
        if advantages.shape != (batch,):
            raise ValueError(f"`advantages` has shape {advantages.shape}, expected {(batch,)}")
        # returns we know is (batch,), by construction



        # === 4) NOW CONVERT TO TENSORS ===
        inputs1         = tf.convert_to_tensor(states, dtype=tf.float32)
        trend_samples  = tf.convert_to_tensor(trend_samples, dtype=tf.float32)
        sample_indices  = tf.convert_to_tensor(sample_indices, dtype=tf.int32)
        logprobs        = tf.convert_to_tensor(logprobs, dtype=tf.float32)
        advantages      = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns         = tf.convert_to_tensor(returns, dtype=tf.float32)



        inputs2 = tf.expand_dims(sample_indices, axis=-1)

        
        # === Critic (Value Network) Update ===
        with tf.GradientTape() as critic_tape:
            new_values, _ = self.value_dict['model'](inputs=(inputs1, inputs2), states=None, training=True)
            new_values = tf.squeeze(new_values)
            critic_loss = tf.reduce_mean(tf.square(returns - new_values))

            tf.debugging.check_numerics(critic_loss, message="Critic loss has NaNs/Infs.")
            tf.debugging.assert_shapes([
                (new_values, returns.shape)
            ])


        # === Param update
        critic_grads = critic_tape.gradient(critic_loss, self.value_dict['model'].trainable_variables)
        self.value_dict['optimizer'].apply_gradients(zip(critic_grads, self.value_dict['model'].trainable_variables))

        # Validate gradients
        for i, g in enumerate(critic_grads):
            if g is None:
                raise ValueError(f"Gradient {i} is None for critic network")
            tf.debugging.check_numerics(g, message=f"Critic gradient {i} has NaNs/Infs")

        if any(tf.reduce_any(tf.math.is_inf(g)) or tf.reduce_any(tf.math.is_nan(g)) for g in critic_grads if g is not None):
            raise ValueError("Exploding or NaN gradients detected in critic")





        

        # === Actor (Policy Network) Update ===
        trend_samples = tf.expand_dims(trend_samples, axis=-1) # [batchsize, 1]
        with tf.GradientTape() as actor_tape:
            mu, sigma, _ = self.policy_dict['model'](inputs=(inputs1, inputs2), states=None, training=True)
            
            # continuous probs
            dist = tfp.distributions.Normal(loc=mu, scale=sigma)
            new_logprobs = tf.squeeze(dist.log_prob(trend_samples)) # (batchsize, 1) sqieezed
            
            

            ratio = tf.exp(new_logprobs - logprobs)
            clipped_ratio = tf.clip_by_value(ratio,
                                                clip_value_min=1.0 - self.clip_pram,
                                                clip_value_max=1.0 + self.clip_pram)
            
            actor_loss = tf.reduce_mean(-tf.minimum(ratio * advantages, clipped_ratio * advantages))


            tf.debugging.check_numerics(actor_loss, message="Actor loss has NaNs/Infs.")


            # === surrogate loss
            tf.debugging.assert_shapes([
                (logprobs, new_logprobs.shape),
                (ratio, advantages.shape)
            ])


        actor_grads = actor_tape.gradient(actor_loss, self.policy_dict['model'].trainable_variables)
        self.policy_dict['optimizer'].apply_gradients(zip(actor_grads, self.policy_dict['model'].trainable_variables))

        if None in actor_grads:
            raise ValueError("NaN or None detected in actor gradients")
        
        # Validate gradients
        for i, g in enumerate(actor_grads):
            if g is None:
                raise ValueError(f"Gradient {i} is None for actor network")
            tf.debugging.check_numerics(g, message=f"Actor gradient {i} has NaNs/Infs")
        
        
        return actor_loss, critic_loss
        
            
            
            


    def validate_hidden_state(self, data):
        if not isinstance(data, list):
            raise ValueError("Data must be a list")
        for i, inner in enumerate(data):
            if not isinstance(inner, list):
                raise ValueError(f"Element at index {i} is not a list")
            for j, tensor in enumerate(inner):
                if not isinstance(tensor, tf.Tensor):
                    raise ValueError(f"Element at [{i}][{j}] is not a TensorFlow tensor")




    
    def learn(self,n_games: int) -> bool:
        """
        Trains the agent for a specified number of epochs.

        Args:
            n_games (int): The number of epochs to train for.

        Returns:
            bool: True if training completed successfully.
        """
        

        n_trajectory_trained = 0
        step_count = 0
        minibatch_count = 0
        total_critic_loss = 1000
        RESET_ENV = False
        self.memory.clear()
        
        # Ensure environment pool is valid
        if not hasattr(self, 'pool') or self.pool is None:
            print("Error: Environment pool is not initialized.")
            return False
        
        while self.epoch_count < n_games:

            # Increase batchsize
            if RESET_ENV == True and self.batch_inc_freq != 0:
                self.pool.close()
                env = self.pool.get_random_env(batch_inc_flag=True)
                RESET_ENV = False
            else:
                env = self.pool.get_random_env(batch_inc_flag=False)

            # close previous env and make new env with fresh data
            
            obs, _ = env.reset(
                options={
                    "channel_weights": self.channel_weights
                }
            )

            policy_states, value_states = None, None


            while True:
                # 1. Make input samples
                input1 = tf.convert_to_tensor([obs], dtype=tf.float32)
                input2 = tf.convert_to_tensor([[i] for i in range(self.signal_size)], dtype=tf.int32)

                # 2. Action on each timestep
                approaches = [
                    self.choose_action(
                        input1=input1,
                        input2=input2[i:i+1],
                        policy_states=policy_states,
                        value_states=value_states
                    ) for i in range(self.signal_size)
                ]


                # 3. Extract results
                trend_samples = np.array([a['sample'].numpy() for a in approaches], dtype=np.float32)
                logprobs = np.array([a['logprob'].numpy() for a in approaches], dtype=np.float32)
                values = np.array([a['value'].numpy() for a in approaches], dtype=np.float32)
                mus = np.array([a['mu'].numpy() for a in approaches], dtype=np.float32)
                sigma = np.array([a['sigma'].numpy() for a in approaches], dtype=np.float32)
                ps = [a['policy_states'] for a in approaches]
                vs = [a['value_states'] for a in approaches]

                # readonly
                trend_samples.flags.writeable = False


        
                # 4. Take step with current action, intencity enters as raw value then altered within stock world->helpers
                next_obs, reward, terminated, truncated, info = env.step(trend_samples)
                best_index = info['best_index']
                # channel that done the best, which will be later utilized for internal state detection (norm_shifts)

                if info["norm_shifts"].shape == mus.shape and np.issubdtype(info["norm_shifts"].dtype, np.floating):
                    self.channel_scores[np.sign(info["norm_shifts"]) == np.sign(mus)] += 1
                else:
                    raise ValueError("Some issue")




                # Hidden state
                policy_states = ps[self.internal_state_index]
                value_states = vs[self.internal_state_index]

                # Validate variables are ok
                self.validate_hidden_state(policy_states)
                self.validate_hidden_state(value_states)

                # keys is related to array name of memory class


                # =================================================================================
                # On Egde case collect value from next obs index 4
                # =================================================================================
                next_val = values[best_index+1] if (best_index < self.signal_size - 1) else 0.0

                if next_val == 0.0:
                    # next value by calling value model
                    next_input1 = tf.convert_to_tensor([next_obs], dtype=tf.float32)
                    next_input2 = tf.convert_to_tensor([[self.signal_size - 1]], dtype=tf.int32)
                    
                    next_val, _ = self.value_dict['model'](
                        inputs=(next_input1, next_input2), 
                        states=value_states, 
                        training=False
                    )
                    next_val = tf.squeeze(next_val).numpy()

                    
                curr_val = values[best_index]
                
                self.memory.store(
                    states=obs,
                    samples=trend_samples[best_index], # action
                    rewards=reward,
                    indices=best_index,
                    logprobs=logprobs[best_index],
                    values=(curr_val, next_val),
                    dones=terminated,
                    truncates=truncated
                )

                
                # 3. Proceed to next step
                step_count += 1
                obs = next_obs

                
                with self.summary_writer2.as_default():
                    tf.summary.scalar('trend', data=trend_samples[best_index], step=step_count)
                with self.summary_writer3.as_default():
                    tf.summary.scalar('trend', data=info['actual_signal'], step=step_count)

                with self.summary_writer5.as_default():
                    tf.summary.scalar('sigma', data=sigma[best_index], step=step_count)
                with self.summary_writer6.as_default():
                    tf.summary.scalar('reward', data=reward, step=step_count)
                    tf.summary.scalar('actual_raw_signals', data=info['actual_raw_signals'][best_index], step=step_count)

                with self.summary_writer7.as_default():
                    tf.summary.scalar('batchsize', data=self.pool.get_batchsize(), step=step_count)
                
                    


                # 4. End of balanced episode
                if truncated:
                    minibatch_count += 1
                    # Reservoir full for training
                    if minibatch_count % self.n_batches == 0:

                        print(f"training... {minibatch_count}")


                        # ... Training code
                        payload = self.memory.get_batches()
                        num_minibatches = len(payload)
                        


                        # =================================================================================
                        # K epoch training
                        # =================================================================================
                        total_critic_loss, total_actor_loss, adv_log = 0.0, 0.0, 0.0
                        # 4. preform training on collection
                        for _ in range(self.k_epochs):
                            group_indices = np.random.permutation(num_minibatches)
                            for idx in group_indices:
                                minibatch = payload[idx]
                                actor_loss, critic_loss = self.train_step(**minibatch)

                                adv_log += float(np.mean(minibatch["advantages"]))
                                total_critic_loss += float(critic_loss.numpy())
                                total_actor_loss += float(actor_loss.numpy())





                        # 5. get avarage loss of all batches
                        num_batches = len(payload)
                        critic_loss = total_critic_loss / num_batches
                        avarage_actor_loss = total_actor_loss / num_batches



                        # =================================================================================
                        # Common index to be used for next hidden state selection
                        # =================================================================================
                        self.internal_state_index = np.argmax(self.channel_scores) # select where it made best progress

                        ts_int, ts_count = np.unique(np.sort(np.array(self.memory.storage["indices"], dtype=np.int32)), return_counts=True)
                        

                        # Check for out-of-bound values
                        if np.any((ts_int < 0) | (ts_int > (self.signal_size - 1))):
                            raise ValueError(f"Found ts_int values out of bound (0-4): {ts_int}")
                        
                        # Create a fixed array with counts for ts_int = 0..4
                        for ts_idx, current_ts_count in zip(ts_int, ts_count):
                            self.channel_weights[ts_idx] += current_ts_count


                        # Returns sum
                        np.add.at(self.returns_weights, self.memory.storage["indices"], self.memory.returns)



                        # Log to TensorBoard as multiple scalars on a single chart
                        with self.sum_writer0.as_default():
                            tf.summary.scalar("count", self.channel_weights[0], step=n_trajectory_trained)
                            tf.summary.scalar("sum_reward", self.returns_weights[0], step=n_trajectory_trained)
                        with self.sum_writer1.as_default():
                            tf.summary.scalar("count", self.channel_weights[1], step=n_trajectory_trained)
                            tf.summary.scalar("sum_reward", self.returns_weights[1], step=n_trajectory_trained)
                        with self.sum_writer2.as_default():
                            tf.summary.scalar("count", self.channel_weights[2], step=n_trajectory_trained)
                            tf.summary.scalar("sum_reward", self.returns_weights[2], step=n_trajectory_trained)
                        with self.sum_writer3.as_default():
                            tf.summary.scalar("count", self.channel_weights[3], step=n_trajectory_trained)
                            tf.summary.scalar("sum_reward", self.returns_weights[3], step=n_trajectory_trained)
                        with self.sum_writer4.as_default():
                            tf.summary.scalar("count", self.channel_weights[4], step=n_trajectory_trained)
                            tf.summary.scalar("sum_reward", self.returns_weights[4], step=n_trajectory_trained)



                        # Other graphs
                        with self.summary_writer0.as_default():
                            tf.summary.scalar('critic_loss', data=critic_loss, step=n_trajectory_trained)
                        with self.summary_writer1.as_default():
                            tf.summary.scalar('actor_loss', data=avarage_actor_loss, step=n_trajectory_trained)
                            tf.summary.scalar('advantages', data=adv_log, step=n_trajectory_trained)
                        with self.summary_writer8.as_default():
                            tf.summary.scalar('common_timesteps', data=self.internal_state_index, step=n_trajectory_trained)
                        
                        # Free up
                        self.memory.clear()
                        n_trajectory_trained += 1
                        # ... Training Finished, memory cleared, prepare for fresh trejectory collection ...
                        if self.batch_inc_freq != 0 and n_trajectory_trained % self.batch_inc_freq == 0:
                            RESET_ENV = True

                        

                    self.epoch_count += 1
                    break


        
        # Clean up
        self.pool.close()


        return total_critic_loss, self.internal_state_index



                