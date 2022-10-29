🚂 Starter Kit - NeurIPS 2020 Flatland Challenge
===

This starter kit contains 2 example policies to get started with this challenge: 
- a simple single-agent DQN method
- a more robust multi-agent DQN method that you can submit out of the box to the challenge 🚀

**🔗 [Train the single-agent DQN policy](https://flatland.aicrowd.com/getting-started/rl/single-agent.html)**

**🔗 [Train the multi-agent DQN policy](https://flatland.aicrowd.com/getting-started/rl/multi-agent.html)**

**🔗 [Submit a trained policy](https://flatland.aicrowd.com/getting-started/first-submission.html)**

The single-agent example is meant as a minimal example of how to use DQN. The multi-agent is a better starting point to create your own solution.

You can fully train the multi-agent policy in Colab for free! [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GbPwZNQU7KJIJtilcGBTtpOAD3EabAzJ?usp=sharing)

Sample training usage
---

Train the multi-agent policy for 150 episodes:

```bash
python reinforcement_learning/multi_agent_training.py -n 150
```

The multi-agent policy training can be tuned using command-line arguments:

```console 
usage: multi_agent_training.py [-h] [-n N_EPISODES] [-t TRAINING_ENV_CONFIG]
                               [-e EVALUATION_ENV_CONFIG]
                               [--n_evaluation_episodes N_EVALUATION_EPISODES]
                               [--checkpoint_interval CHECKPOINT_INTERVAL]
                               [--eps_start EPS_START] [--eps_end EPS_END]
                               [--eps_decay EPS_DECAY]
                               [--buffer_size BUFFER_SIZE]
                               [--buffer_min_size BUFFER_MIN_SIZE]
                               [--restore_replay_buffer RESTORE_REPLAY_BUFFER]
                               [--save_replay_buffer SAVE_REPLAY_BUFFER]
                               [--batch_size BATCH_SIZE] [--gamma GAMMA]
                               [--tau TAU] [--learning_rate LEARNING_RATE]
                               [--hidden_size HIDDEN_SIZE]
                               [--update_every UPDATE_EVERY]
                               [--use_gpu USE_GPU] [--num_threads NUM_THREADS]
                               [--render RENDER]

optional arguments:
  -h, --help            show this help message and exit
  -n N_EPISODES, --n_episodes N_EPISODES
                        number of episodes to run
  -t TRAINING_ENV_CONFIG, --training_env_config TRAINING_ENV_CONFIG
                        training config id (eg 0 for Test_0)
  -e EVALUATION_ENV_CONFIG, --evaluation_env_config EVALUATION_ENV_CONFIG
                        evaluation config id (eg 0 for Test_0)
  --n_evaluation_episodes N_EVALUATION_EPISODES
                        number of evaluation episodes
  --checkpoint_interval CHECKPOINT_INTERVAL
                        checkpoint interval
  --eps_start EPS_START
                        max exploration
  --eps_end EPS_END     min exploration
  --eps_decay EPS_DECAY
                        exploration decay
  --buffer_size BUFFER_SIZE
                        replay buffer size
  --buffer_min_size BUFFER_MIN_SIZE
                        min buffer size to start training
  --restore_replay_buffer RESTORE_REPLAY_BUFFER
                        replay buffer to restore
  --save_replay_buffer SAVE_REPLAY_BUFFER
                        save replay buffer at each evaluation interval
  --batch_size BATCH_SIZE
                        minibatch size
  --gamma GAMMA         discount factor
  --tau TAU             soft update of target parameters
  --learning_rate LEARNING_RATE
                        learning rate
  --hidden_size HIDDEN_SIZE
                        hidden size (2 fc layers)
  --update_every UPDATE_EVERY
                        how often to update the network
  --use_gpu USE_GPU     use GPU if available
  --num_threads NUM_THREADS
                        number of threads PyTorch can use
  --render RENDER       render 1 episode in 100
```

[**📈 Performance training in environments of various sizes**](https://wandb.ai/masterscrat/flatland-examples-reinforcement_learning/reports/Flatland-Starter-Kit-Training-in-environments-of-various-sizes--VmlldzoxNjgxMTk)

[**📈 Performance with various hyper-parameters**](https://app.wandb.ai/masterscrat/flatland-examples-reinforcement_learning/reports/Flatland-Examples--VmlldzoxNDI2MTA)

[![](https://i.imgur.com/Lqrq5GE.png)](https://app.wandb.ai/masterscrat/flatland-examples-reinforcement_learning/reports/Flatland-Examples--VmlldzoxNDI2MTA) 

Credits
---

* Florian Laurent <florian@aicrowd.com>
* Erik Nygren <erik.nygren@sbb.ch>
* Adrian Egli <adrian.egli@sbb.ch>
* Sharada Mohanty <mohanty@aicrowd.com>
* Christian Baumberger <christian.baumberger@sbb.ch>
* Guillaume Mollard <guillaume.mollard2@gmail.com>

Main links
---

* [Flatland documentation](https://flatland.aicrowd.com/)