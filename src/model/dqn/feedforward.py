import torch
import torch.nn as nn
from torch.autograd import Variable
from .base import DQNModuleBase, DQN


class DQNModuleFeedforward(DQNModuleBase):

    def __init__(self, params):
        super(DQNModuleFeedforward, self).__init__(params)

        self.feedforward = nn.Sequential(
            nn.Linear(self.output_dim, params.hidden_dim),
            nn.Sigmoid())

    def forward(self, x_screens, x_variables):
        """
        Argument sizes:
            - x_screens of shape (batch_size, seq_len * n_fm, h, w)
            - x_variables list of n_var tensors of shape (batch_size,)
        """

        batch_size = x_screens.size(0)
        assert x_screens.ndimension() == 4
        assert len(x_variables) == self.n_variables
        assert all(x.ndimension() == 1 and x.size(0) == batch_size
                   for x in x_variables)

        # state input (screen / depth / labels buffer + variables)
        state_input, output_gf = self.base_forward(x_screens, x_variables)

        # apply the feed forward middle
        state_input = self.feedforward(state_input)

        # apply the head to feed forward result
        output_sc = self.head_forward(state_input)

        return output_sc, output_gf


class DQNFeedforward(DQN):

    DQNModuleClass = DQNModuleFeedforward

    def f_eval(self, last_states):

        screens, variables = self.prepare_f_eval_args(last_states)

        return self.module(
            screens.view(1, -1, *self.screen_shape[1:]),
            [variables[-1, i] for i in range(self.params.n_variables)]
        )

    def f_train(self, screens, variables, features, actions, rewards, isfinal,
                loss_history=None):

        screens, variables, features, actions, rewards, isfinal = \
            self.prepare_f_train_args(screens, variables, features,
                                      actions, rewards, isfinal)

        batch_size = self.params.batch_size
        seq_len = self.hist_size + 1

        screens = screens.view(batch_size, seq_len * self.params.n_fm,
                               *self.screen_shape[1:])

        output_sc1, output_gf1 = self.module(
            screens[:, :-self.params.n_fm, :, :],
            [variables[:, -2, i] for i in range(self.params.n_variables)]
        )
        output_sc2, output_gf2 = self.module(
            screens[:, self.params.n_fm:, :, :],
            [variables[:, -1, i] for i in range(self.params.n_variables)]
        )

        # compute scores
        mask = torch.ByteTensor(output_sc1.size()).fill_(0)
        for i in range(batch_size):
            mask[i, int(actions[i, -1])] = 1
        scores1 = output_sc1.masked_select(self.get_var(mask))
        scores2 = rewards[:, -1] + (
            self.params.gamma * output_sc2.max(1)[0] * (1 - isfinal[:, -1])
        )

        # dqn loss
        loss_sc = self.loss_fn_sc(scores1, Variable(scores2.data))

        # game features loss
        loss_gf = 0
        if self.n_features:
            loss_gf += self.loss_fn_gf(output_gf1, features[:, -2].float())
            loss_gf += self.loss_fn_gf(output_gf2, features[:, -1].float())

        self.register_loss(loss_history, loss_sc, loss_gf)

        return loss_sc, loss_gf

    @staticmethod
    def validate_params(params):
        DQN.validate_params(params)
        assert params.recurrence == ''
