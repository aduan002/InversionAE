import os
import random
import numpy as np
import math
import shutil

class SampleDistribution:
    # Theory Reference: https://statisticsbyjim.com/hypothesis-testing/sampling-distribution/
    def __init__(self, sample_size = 30, num_samples = 500) -> None:
        self.sample_size = sample_size
        self.num_samples = num_samples

    def sample(self, files):
        return random.choices(files, k = self.sample_size)
    
    def caculate_sample_mean(self, file_dir, file_names):
        # NOTE: There is a faster way to do this, but it takes a non-constant space complexity, so I went with this instead.
        mean = None
        num_files = 0
        for file in file_names:
            sample = np.loadtxt(os.path.join(file_dir, file)) # NOTE: Assumes no missing values

            if mean is None:
                mean = np.zeros(shape=sample.shape)

            np.add(mean, sample, out=mean)
            num_files += 1
        return mean / num_files

    
    def calculate_sampling_distribution(self, file_dir, file_names):
        sampling_means = None
        for i in range(self.num_samples): # NOTE: Issue here where a large number of samples take a large amount of memory, should do streaming mean and std calculations instead.
            single_sample = self.sample(file_names)
            sample_mean = self.caculate_sample_mean(file_dir, single_sample)

            if sampling_means is None:
                sampling_means = np.zeros(shape=(self.num_samples, sample_mean.shape[0]))

            sampling_means[i] = sample_mean
        mean = np.mean(sampling_means, axis = 0)
        std = np.std(sampling_means, axis = 0)

        return mean, std
    
    def calculate_parent_distribution(self, file_dir, file_names):
        # The parent distribution is mu and sigma
        # The sampling distribution is mu and sigma / sqrt(n)
        sample_mean, sample_std = self.calculate_sampling_distribution(file_dir, file_names)
        return sample_mean, sample_std * math.sqrt(self.sample_size)


class SplitData:
    def __init__(self, input_dir, output_dir) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def find_split(self, distribution, train_split = 0.7, val_split = 0.2, test_split = 0.1, random_state = -1, max_iter = 1000):
        if random_state != -1:
            random.seed(random_state)

        mean, std = distribution
        lower_bound = mean - std
        upper_bound = mean + std

        file_names = os.listdir(self.input_dir)
        sampler = SampleDistribution()

        train_idx = int(train_split * len(file_names))
        val_idx = train_idx + int(val_split * len(file_names))
        test_idx = val_idx + int(test_split * len(file_names))

        good_distribution = False

        best_train_split = None
        best_val_split = None
        best_test_split = None
        bestScore = None

        iter = 1
        while not good_distribution and iter <= max_iter:
            print("iter:", iter)
            random.shuffle(file_names)

            train_split = file_names[:train_idx]
            val_split = file_names[train_idx:val_idx]
            test_split = file_names[val_idx:test_idx]

            train_mean = sampler.caculate_sample_mean(self.input_dir, train_split)
            val_mean = sampler.caculate_sample_mean(self.input_dir, val_split)
            test_mean = sampler.caculate_sample_mean(self.input_dir, test_split)

            # The idea is that we take the absolute value of every delta with respect to the original mean and add them to a total score, we lower better.
            # Not doing average because it would simply chnage the score by a constant factor, but this can be thought of as taking average wherever, too.
            score = np.sum( np.absolute(np.subtract(train_mean, mean)) + np.absolute(np.subtract(val_mean, mean)) + np.absolute(np.subtract(test_mean, mean)) )
            
            # Since score is a sum of delta, the lower the better
            if bestScore is None or score < bestScore:
                bestScore = score
                best_train_split = train_split
                best_val_split = val_split
                best_test_split = test_split

            # If all the sample means fall within one std of the parent mean.
            if np.less_equal(lower_bound, train_mean).all() and np.less_equal(train_mean, upper_bound).all() and \
               np.less_equal(lower_bound, val_mean).all() and np.less_equal(val_mean, upper_bound).all() and \
               np.less_equal(lower_bound, test_mean).all() and np.less_equal(test_mean, upper_bound).all():
                good_distribution = True

            # Increment random state in a deterministic way for reproducible results.
            if random_state != -1:
                random.seed(random_state + iter)
            iter += 1

        if not good_distribution: # The only way this is true is if we hit the max iter before finding a good distribution.
            train_split = best_train_split
            val_split = best_val_split
            test_split = best_test_split

        return train_split, val_split, test_split
    
    def create_split(self, distribution, train_split = 0.7, val_split = 0.2, test_split = 0.1, random_state = -1, max_iter = 1000):
        train_split, val_split, test_split = self.find_split(distribution, train_split, val_split, test_split, random_state, max_iter)

        train_dir = os.path.join(self.output_dir, "train")
        val_dir = os.path.join(self.output_dir, "val")
        test_dir = os.path.join(self.output_dir, "test")

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        for file in train_split:
            file_src = os.path.join(self.input_dir, file)
            file_dst = os.path.join(train_dir, file)
            shutil.copy(file_src, file_dst)
        for file in val_split:
            file_src = os.path.join(self.input_dir, file)
            file_dst = os.path.join(val_dir, file)
            shutil.copy(file_src, file_dst)
        for file in test_split:
            file_src = os.path.join(self.input_dir, file)
            file_dst = os.path.join(test_dir, file)
            shutil.copy(file_src, file_dst)


if __name__ == "__main__":
    input_dir = "filtered_sigmas"
    output_dir = "filtered_data_TEST"

    file_names = os.listdir(input_dir)
    sampler = SampleDistribution()
    distribution = sampler.calculate_parent_distribution(input_dir, file_names)

    splitter = SplitData(input_dir, output_dir)
    splitter.create_split(distribution, random_state = 42)


