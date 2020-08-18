""" 
Cyclic learning rate scheduling.
Adapted from https://arxiv.org/pdf/1811.11431.pdf
"""

class MyLRScheduler(object):
    def __init__(self, max_lr=0.5, cycle_len=5, warm_up_interval=1):
        super(MyLRScheduler, self).__init__()
        self.min_lr = max_lr / cycle_len # minimum learning rate
        self.max_lr = max_lr  # maximum learning rate
        self.cycle_len = cycle_len
        self.warm_up_interval = warm_up_interval # we do not start from max value for the first epoch, because some time it diverges
        self.counter = 0
        print('Using Cyclic LR Scheduler with warm restarts')

    def get_lr(self, epoch):
        current_lr = self.min_lr
        # warm-up or cool-down phase
        if self.counter < self.warm_up_interval:
            self.counter += 1
            # Apply warm up only once
            if self.counter == self.warm_up_interval:
                self.warm_up_interval = 0
        else:
            #Cyclic learning rate with warm restarts
            # max_lr (= min_lr * step_size) is decreased to min_lr using linear decay before
            # it is set to max value at the end of cycle.
            current_lr = round(self.max_lr - (epoch % self.cycle_len) * self.min_lr, 5)

        return current_lr


if __name__ == '__main__':
    lrSched_60 = MyLRScheduler(max_lr=0.3, cycle_len=5, warm_up_interval=2)
    lrSched_120 = MyLRScheduler(max_lr=0.1, cycle_len=60, warm_up_interval=0)
    for i in range(0,60):
        print(i+1, lrSched_60.get_lr(i))
    for i in range(60,120):
        print(i+1, lrSched_120.get_lr(i))