from copy import deepcopy
import torch.multiprocessing as mp


class ParameterServer(object):

    def __init__(self, n_processes):
        self.queue = mp.Queue()
        self.lock = mp.Lock()
        self.n_processes = n_processes
        self.params = None

    def __getstate__(self):
        return (self.queue, self.lock)

    def __setstate__(self, state):
        self.queue, self.lock = state

    def set_rank(self, rank):
        self.rank = rank

    def register_model(self, model):
        if self.rank == 0:
            self.params = deepcopy(model.state_dict())
            # FIXME: this should be fixed in Variable multiprocessing reducer
            # We have to touch the grads to ensure they're going to be shared
            # TODO: needs urgent fix
            # for param in self.params.values():
            #     param.grad
            for i in range(self.n_processes - 1):
                self.queue.put(self.params)
        else:
            self.params = self.queue.get()
            assert set(model.state_dict().keys()) == set(self.params.keys())

    @property
    def state_dict(self):
        return self.params
