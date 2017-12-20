import torch
from torch.autograd import Variable
from .base import DQNModuleBase, DQN
from ..utils import get_recurrent_module
from ...utils import bool_flag


class DQNModuleRecurrent(DQNModuleBase):

    def __init__(self, params):
        super(DQNModuleRecurrent, self).__init__(params)

        recurrent_module = get_recurrent_module(params.recurrence)
        self.rnn = recurrent_module(self.output_dim, params.hidden_dim,
                                    num_layers=params.n_rec_layers,
                                    dropout=params.dropout,
                                    batch_first=True)

    def forward(self, x_screens, x_variables, prev_state):
        """
        Argument sizes:
            - x_screens of shape (batch_size, seq_len, n_fm, h, w)
            - x_variables list of n_var tensors of shape (batch_size, seq_len)
        """
        batch_size = x_screens.size(0)
        seq_len = x_screens.size(1)

        assert x_screens.ndimension() == 5
        assert len(x_variables) == self.n_variables
        assert all(x.ndimension() == 2 and x.size(0) == batch_size and
                   x.size(1) == seq_len for x in x_variables)

        # We're doing a batched forward through the network base
        # Flattening seq_len into batch_size ensures that it will be applied
        # to all timesteps independently.
        state_input, output_gf = self.base_forward(
            x_screens.view(batch_size * seq_len, *x_screens.size()[2:]),
            [v.contiguous().view(batch_size * seq_len) for v in x_variables]
        )

        # unflatten the input and apply the RNN
        rnn_input = state_input.view(batch_size, seq_len, self.output_dim)
        rnn_output, next_state = self.rnn(rnn_input, prev_state)
        rnn_output = rnn_output.contiguous()

        # apply the head to RNN hidden states (simulating larger batch again)
        output_sc = self.head_forward(rnn_output.view(-1, self.hidden_dim))

        # unflatten scores and game features
        output_sc = output_sc.view(batch_size, seq_len, output_sc.size(1))
        if self.n_features:
            output_gf = output_gf.view(batch_size, seq_len, self.n_features)

        return output_sc, output_gf, next_state


class DQNRecurrent(DQN):

    DQNModuleClass = DQNModuleRecurrent

    def __init__(self, params):
        super(DQNRecurrent, self).__init__(params)
        h_0 = torch.FloatTensor(params.n_rec_layers, params.batch_size,
                                params.hidden_dim).zero_()
        self.init_state_t = self.get_var(h_0)
        self.init_state_e = Variable(self.init_state_t[:, :1, :].data.clone(), volatile=True)
        if params.recurrence == 'lstm':
            self.init_state_t = (self.init_state_t, self.init_state_t)
            self.init_state_e = (self.init_state_e, self.init_state_e)
        self.reset()

    def reset(self):
        # prev_state is only used for evaluation, so has a batch size of 1
        self.prev_state = self.init_state_e

    def f_eval(self, last_states):

        screens, variables = self.prepare_f_eval_args(last_states)

        # if we remember the whole sequence, only feed the last frame
        if self.params.remember:
            output = self.module(
                screens[-1:].view(1, 1, *self.screen_shape),
                [variables[-1:, i].view(1, 1)
                 for i in range(self.params.n_variables)],
                prev_state=self.prev_state
            )
            # save the hidden state if we want to remember the whole sequence
            self.prev_state = output[-1]
        # otherwise, feed the last `hist_size` ones
        else:
            output = self.module(
                screens.view(1, self.hist_size, *self.screen_shape),
                [variables[:, i].contiguous().view(1, self.hist_size)
                 for i in range(self.params.n_variables)],
                prev_state=self.prev_state
            )

        # do not return the recurrent state
        return output[:-1]

    def f_train(self, screens, variables, features, actions, rewards, isfinal,
                loss_history=None):

        screens, variables, features, actions, rewards, isfinal = \
            self.prepare_f_train_args(screens, variables, features,
                                      actions, rewards, isfinal)

        batch_size = self.params.batch_size
        seq_len = self.hist_size + self.params.n_rec_updates

        output_sc, output_gf, _ = self.module(
            screens,
            [variables[:, :, i] for i in range(self.params.n_variables)],
            prev_state=self.init_state_t
        )

        # compute scores
        mask = torch.ByteTensor(output_sc.size()).fill_(0)
        for i in range(batch_size):
            for j in range(seq_len - 1):
                mask[i, j, int(actions[i, j])] = 1
        scores1 = output_sc.masked_select(self.get_var(mask))
        scores2 = rewards + (
            self.params.gamma * output_sc[:, 1:, :].max(2)[0] * (1 - isfinal)
        )

        # dqn loss
        loss_sc = self.loss_fn_sc(
            scores1.view(batch_size, -1)[:, -self.params.n_rec_updates:],
            Variable(scores2.data[:, -self.params.n_rec_updates:])
        )

        # game features loss
        if self.n_features:
            loss_gf = self.loss_fn_gf(output_gf, features.float())
        else:
            loss_gf = 0

        self.register_loss(loss_history, loss_sc, loss_gf)

        return loss_sc, loss_gf

    @staticmethod
    def register_args(parser):
        DQN.register_args(parser)
        parser.add_argument("--n_rec_updates", type=int, default=1,
                            help="Number of updates to perform")
        parser.add_argument("--n_rec_layers", type=int, default=1,
                            help="Number of recurrent layers")
        parser.add_argument("--remember", type=bool_flag, default=True,
                            help="Remember the whole sequence")

    @staticmethod
    def validate_params(params):
        DQN.validate_params(params)
        assert params.recurrence in ['rnn', 'gru', 'lstm']
        assert params.n_rec_updates >= 1
        assert params.n_rec_layers >= 1
