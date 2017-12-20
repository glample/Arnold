import torch
import os
import numpy as np
from logging import getLogger

from .utils import get_optimizer
from .replay_memory import ReplayMemory


logger = getLogger()


class Trainer(object):

    def __init__(self, params, game, network, eval_fn, parameter_server=None):
        optim_fn, optim_params = get_optimizer(params.optimizer)
        self.optimizer = optim_fn(network.module.parameters(), **optim_params)
        self.parameter_server = parameter_server
        self.params = params
        self.game = game
        self.network = network
        self.eval_fn = eval_fn
        self.state_dict = self.network.module.state_dict()
        self.n_iter = 0
        self.best_score = -1000000

    def start_game(self):
        map_id = np.random.choice(self.params.map_ids_train)
        logger.info("Training on map %i ..." % map_id)
        self.game.start(map_id=map_id,
                        episode_time=self.params.episode_time,
                        log_events=False,
                        manual_control=False)
        if hasattr(self.params, 'randomize_textures'):
            self.game.randomize_textures(self.params.randomize_textures)
        if hasattr(self.params, 'init_bots_health'):
            self.game.init_bots_health(self.params.init_bots_health)

    def run(self):
        self.start_game()
        self.network.reset()

        network_type = self.params.network_type
        update_frequency = self.params.update_frequency
        log_frequency = self.params.log_frequency
        dump_frequency = self.params.dump_freq

        # log current training loss
        current_loss = self.network.new_loss_history()

        last_states = []
        start_iter = self.n_iter
        last_eval_iter = self.n_iter
        last_dump_iter = self.n_iter

        self.network.module.train()

        while True:
            self.n_iter += 1

            if self.game.is_final():
                self.game.reset()     # dead or end of episode
                self.network.reset()  # reset internal state (RNNs only)

            self.game.observe_state(self.params, last_states)

            # select the next action. `action` will correspond to an action ID
            # if we use non-continuous actions, otherwise it will correspond
            # to a set of continuous / discontinuous actions.
            # if DQN, epsilon greedy or action with the highest score
            random_action = network_type.startswith('dqn') and self.epsilon_greedy()
            if (network_type.startswith('dqn') and
                (not random_action or self.params.recurrence != '')):
                self.network.module.eval()
                action = self.network.next_action(last_states, save_graph=True)
                self.network.module.train()
            if random_action:
                action = np.random.randint(self.params.n_actions)

            # perform the action, and skip some frames
            self.game.make_action(action, self.params.frame_skip)

            # save last screens / features / action
            self.game_iter(last_states, action)

            # evaluation
            if (self.n_iter - last_eval_iter) % self.params.eval_freq == 0:
                self.evaluate_model(start_iter)
                last_eval_iter = self.n_iter

            # periodically dump the model
            if (dump_frequency > 0 and
                    (self.n_iter - last_dump_iter) % dump_frequency == 0):
                self.dump_model(start_iter)
                last_dump_iter = self.n_iter

            # log current average loss
            if self.n_iter % (log_frequency * update_frequency) == 0:
                logger.info('=== Iteration %i' % self.n_iter)
                self.network.log_loss(current_loss)
                current_loss = self.network.new_loss_history()

            train_loss = self.training_step(current_loss)
            if train_loss is None:
                continue

            # backward
            self.optimizer.zero_grad()
            sum(train_loss).backward()
            for p in self.network.module.parameters():
                p.grad.data.clamp_(-5, 5)

            # update
            self.sync_update_parameters()

        self.game.close()

    def game_iter(self, last_states, action):
        raise NotImplementedError

    def training_step(current_loss):
        raise NotImplementedError

    def epsilon_greedy(self):
        """
        For DQN models, return whether we randomly select the next action.
        """
        start_decay = self.params.start_decay
        stop_decay = self.params.stop_decay
        final_decay = self.params.final_decay
        if final_decay == 1:
            return True
        slope = float(start_decay - self.n_iter) / (stop_decay - start_decay)
        p_random = np.clip((1 - final_decay) * slope + 1, final_decay, 1)
        return np.random.rand() < p_random

    def evaluate_model(self, start_iter):
        self.game.close()
        # if we are using a recurrent network, we need to reset the history
        new_score = self.eval_fn(self.game, self.network,
                                 self.params, self.n_iter)
        if new_score > self.best_score:
            self.best_score = new_score
            logger.info('New best score: %f' % self.best_score)
            model_name = 'best-%i.pth' % (self.n_iter - start_iter)
            model_path = os.path.join(self.params.dump_path, model_name)
            logger.info('Best model dump: %s' % model_path)
            torch.save(self.network.module.state_dict(), model_path)
        self.network.module.train()
        self.start_game()

    def dump_model(self, start_iter):
        model_name = 'periodic-%i.pth' % (self.n_iter - start_iter)
        model_path = os.path.join(self.params.dump_path, model_name)
        logger.info('Periodic dump: %s' % model_path)
        torch.save(self.network.module.state_dict(), model_path)

    def sync_update_parameters(self):
        server = self.parameter_server
        if server is None or server.n_processes == 1:
            self.optimizer.step()
            return
        shared_dict = server.state_dict
        grad_scale = 1. / server.n_processes
        if server.rank == 0:
            # accumulate shared gradients into the local copy
            for k in self.state_dict:
                self.state_dict[k].grad.mul_(grad_scale).add_(shared_dict[k].grad)
            # do optimization
            self.optimizer.step()
            # copy updated parameters
            self.sync_dicts(self.state_dict, shared_dict)
            # zero shared gradients
            for v in shared_dict.values():
                v.grad.zero_()
        else:
            # accumulate gradients
            for k in shared_dict:
                shared_dict[k].grad.add_(grad_scale, self.state_dict[k].grad)
            # copy shared parameters
            self.sync_dicts(shared_dict, self.state_dict)

    def sync_dicts(self, src, dst, attr='data'):
        # TODO: use page-locked memory for parameter server
        for k in src:
            getattr(dst[k], attr).copy_(getattr(src[k], attr))


class ReplayMemoryTrainer(Trainer):

    def __init__(self, params, *args, **kwargs):
        super(ReplayMemoryTrainer, self).__init__(params, *args, **kwargs)

        # initialize the replay memory
        self.replay_memory = ReplayMemory(
            params.replay_memory_size,
            (params.n_fm, params.height, params.width),
            params.n_variables, params.n_features
        )

    def game_iter(self, last_states, action):
        # store the transition in the replay table
        self.replay_memory.add(
            screen=last_states[-1].screen,
            variables=last_states[-1].variables,
            features=last_states[-1].features,
            action=action,
            reward=self.game.reward,
            is_final=self.game.is_final()
        )

    def training_step(self, current_loss):
        # enforce update frequency
        if self.n_iter % self.params.update_frequency != 0:
            return
        # prime the training
        if self.replay_memory.size < self.params.batch_size:
            return

        # sample from replay memory and compute predictions and losses
        memory = self.replay_memory.get_batch(
            self.params.batch_size,
            self.params.hist_size + (0 if self.params.recurrence == ''
                                     else self.params.n_rec_updates - 1)
        )
        return self.network.f_train(loss_history=current_loss, **memory)
