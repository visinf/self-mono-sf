from torch.utils.data.sampler import Sampler

class CustomBatchSampler(Sampler):

    def __init__(self, sampler):
        for samp in sampler:
            if not isinstance(samp, Sampler):
                raise ValueError("sampler should be an instance of "
                                 "torch.utils.data.Sampler, but got sampler={}"
                                 .format(samp))
        self.samplers = sampler
        self.n_samples = [len(samp) for samp in self.samplers]
        self.sample_cnt = [0 for samp in self.samplers]
        self.iters = [iter(samp) for samp in self.samplers]

        self.batch_size = [1, 3]

    def __iter__(self):       

        for ii in range(len(self)):

            for ss, samp in enumerate(self.samplers):
                self.sample_cnt[ss] += self.batch_size[ss]
                if self.sample_cnt[ss] > self.n_samples[ss]:
                    self.iters[ss] = iter(samp)
                    self.sample_cnt[ss] = self.batch_size[ss]

            batch = []

            ## for each sampler
            for ss in range(len(self.samplers)):
                if ss is 0:
                    prev_idx = 0
                else:
                    prev_idx = self.n_samples[ss-1]

                for bb in range(self.batch_size[ss]):
                    batch.append(next(self.iters[ss]) + prev_idx)

            yield batch        

    def __len__(self):
        return len(self.samplers[0])
