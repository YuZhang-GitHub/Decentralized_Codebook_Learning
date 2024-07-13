import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as scio

from model_classes import Actor, Critic, OUNoise, init_weights
from environment import envCB


def train(ch, options, train_options, beam_id):
    with torch.cuda.device(options['gpu_idx']):
        print(f'Beam {beam_id}: training begins. GPU being used: {torch.cuda.current_device()}')

        options['ph_table_rep'] = options['ph_table_rep'].cuda()
        options['multi_step'] = options['multi_step'].cuda()
        options['ph_table'] = options['ph_table'].cuda()

        options['H_interf'] = options['H_interf'].cuda()
        options['RF_cb'] = options['RF_cb'].cuda()

        actor_net = Actor(options['num_ant'], options['num_ant'])
        critic_net = Critic(2 * options['num_ant'], 1)
        ounoise = OUNoise((1, options['num_ant']))
        CB_Env = envCB(ch, options['num_ant'], options['num_bits'], beam_id, options)

        actor_net.cuda()
        critic_net.cuda()
        actor_net.apply(init_weights)
        critic_net.apply(init_weights)

        critic_optimizer = optim.Adam(critic_net.parameters(), lr=1e-3)
        actor_optimizer = optim.Adam(actor_net.parameters(), lr=1e-3, weight_decay=1e-2)
        critic_criterion = nn.MSELoss()

        if train_options['overall_iter'] == 1:
            state = torch.zeros((1, options['num_ant'])).float().cuda()  # vector of phases
            print('Initial State Activated.')
        else:
            state = train_options['state']

        # -------------- training -------------- #
        replay_memory = train_options['replay_memory']
        iteration = 0
        num_of_iter = train_options['num_iter']

        sinr_pred_array = np.zeros(num_of_iter)
        bf_gain_pred_array = np.zeros(num_of_iter)

        while iteration < num_of_iter:

            # Proto-action
            actor_net.eval()
            action_pred = actor_net(state)
            reward_pred, bf_gain_pred, sir_pred, action_quant_pred, state_1_pred = CB_Env.get_reward(action_pred)
            reward_pred = torch.from_numpy(reward_pred).float().cuda()

            critic_net.eval()
            q_pred = critic_net(state, action_quant_pred)

            # Exploration and Quantization Processing
            action_pred_noisy = ounoise.get_action(action_pred,
                                                   t=train_options['overall_iter'])  # torch.Size([1, action_dim])
            mat_dist = torch.abs(action_pred_noisy.reshape(options['num_ant'], 1) - options['ph_table_rep'])
            action_quant = options['ph_table_rep'][range(options['num_ant']), torch.argmin(mat_dist, dim=1)].reshape(1,
                                                                                                                     -1)

            # action_quant = action_pred

            state_1, reward, bf_gain, sir, terminal = CB_Env.step(action_quant)
            reward = torch.from_numpy(reward).float().cuda()
            action = action_quant.reshape((1, -1)).float().cuda()

            replay_memory.append((state, action, reward, state_1, terminal))
            replay_memory.append((state, action_quant_pred, reward_pred, state_1_pred, terminal))
            while len(replay_memory) > train_options['replay_memory_size']:
                replay_memory.pop(0)  # clear the oldest memory

            # -------------- Experience Replay -------------- #

            minibatch = random.sample(replay_memory, min(len(replay_memory), train_options['minibatch_size']))

            state_batch = torch.cat(tuple(d[0] for d in minibatch))
            action_batch = torch.cat(tuple(d[1] for d in minibatch))
            reward_batch = torch.cat(tuple(d[2] for d in minibatch))
            state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

            state_batch = state_batch.detach()
            action_batch = action_batch.detach()
            reward_batch = reward_batch.detach()
            state_1_batch = state_1_batch.detach()

            if torch.cuda.is_available():
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()
                state_1_batch = state_1_batch.cuda()

            # loss calculation for Critic Network
            critic_net.train()
            Q_prime = reward_batch
            Q_pred = critic_net(state_batch, action_batch)
            critic_loss = critic_criterion(Q_pred, Q_prime.detach())

            # Update Critic Network
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # loss calculation for Actor Network
            actor_net.train()
            critic_net.eval()
            actor_loss = torch.mean(-critic_net(state_batch, actor_net(state_batch)))

            # Update Actor Network
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = state_1
            train_options['overall_iter'] += 1

            if train_options['overall_iter'] % options['save_freq'] == 0:
                if not os.path.exists('pretrained_model/'):
                    os.mkdir('pretrained_model/')
                PATH = 'pretrained_model/beam' + str(beam_id) + '_iter' + str(train_options['overall_iter']) + '.pth'
                torch.save(critic_net.state_dict(), PATH)
                torch.save(actor_net.state_dict(), PATH)

            if iteration == 0:
                sinr_pred_array[iteration] = max(sir_pred, sir)
                bf_gain_pred_array[iteration] = max(bf_gain_pred, bf_gain)
                sir_pred_best = max(sir_pred, sir)
                bf_gain_pred_best = max(bf_gain_pred, bf_gain)
            else:
                sinr_pred_array[iteration] = max(sir_pred, sir, sir_pred_best)
                if sir_pred_best < max(sir_pred, sir):
                    sir_pred_best = max(sir_pred, sir)
                bf_gain_pred_array[iteration] = max(bf_gain_pred, bf_gain, bf_gain_pred_best)
                if bf_gain_pred_best < max(bf_gain_pred, bf_gain):
                    bf_gain_pred_best = max(bf_gain_pred, bf_gain)

            iteration += 1

            if iteration % 100 == 0:
                _, ax = plt.subplots(1, figsize=(8, 4))
                ax.plot(range(iteration), sinr_pred_array[:iteration], '-k', alpha=1, label='SINR')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('SINR')
                ax.grid(True)
                ax.legend()

                ax1 = ax.twinx()
                ax1.plot(range(iteration), bf_gain_pred_array[:iteration], '-b', alpha=1, label='BF Gain')
                ax1.set_ylabel('BF gain')
                ax1.legend(loc=1)

                plt.draw()
                plt.savefig(os.path.join(options['stats_path'], 'learning_curve.png'))
                
                plt.close()

            print(
                "Beam: %d, Iter: %d, Q: %.4f, Reward: %.2f, BF pred: %.2f, BF: %.2f, SIR pred: %.2f, SIR: %.2f, Critic Loss: %.2f, Policy Loss: %.2f" % \
                (beam_id, train_options['overall_iter'],
                 torch.Tensor.cpu(q_pred.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(reward_pred).numpy().squeeze(),
                 torch.Tensor.cpu(bf_gain_pred.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(bf_gain.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(sir_pred.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(sir.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(critic_loss.detach()).numpy().squeeze(),
                 torch.Tensor.cpu(actor_loss.detach()).numpy().squeeze()))

        train_options['replay_memory'] = replay_memory
        train_options['state'] = state
        train_options['best_state'] = CB_Env.best_bf_vec

    return train_options
