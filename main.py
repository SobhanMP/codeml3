import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torchaudio
import torchaudio.functional as Fa
from random import random
import numpy as np
import pandas as pd
from adabelief_pytorch import AdaBelief
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score
import wandb
from tqdm.auto import tqdm
import argparse
import tempfile

wandb.login()
def replicate(x, xn):
    while xn.shape[1] <= x.shape[1]:
        xn = th.cat([xn, xn], dim=1)
    if xn.shape[1] > x.shape[1]:
        xn = xn[:, :x.shape[1]]
    return xn


class Loader:
    def __init__(self, sample_rate, beta, df, noise_df, sigma, coinflip):
        self.df = df
        self.noise_df = noise_df
        self.sigma = sigma
        self.beta = beta
        self.coinflip = coinflip
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.df)

    def f(self, r):
        audio, rate = torchaudio.load(f'data/fold{r.fold}/{r.slice_file_name}')
        if audio.shape[0] == 1:
            audio = th.stack([audio[0, :], audio[0, :]], dim=0)
        y = th.tensor(float(r.Label))
        return Fa.resample(audio, rate, self.sample_rate), y

    def __getitem__(self, i):
        assert 0 <= i < len(self)
        r = self.df.iloc[i]
        x, y = self.f(r)

        if random() <= self.coinflip and self.noise_df is not None:
            xn = self.f(self.noise_df.sample(n=1).iloc[0])[0]
            xn = replicate(x, xn)
            x = x + np.random.uniform(self.beta, self.beta) * x.norm(2) / xn.norm(2) * xn 
        if random() <= self.coinflip and self.sigma > 0:
            x = x + self.randn(x.shape) * self.sigma * x.norm(1+1)
        return x, y


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = nn.utils.rnn.pad_sequence(batch,
                                      batch_first=True, padding_value=0.)
    return batch


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors.append(waveform)
        targets.append(label)

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = th.stack(targets)

    return tensors, targets


def make_loader(sample_rate, beta,df, batch_size, drop_last=False,
                shuffle=False,
                noise_df=None,
                num_workers=8,
                coinflip=1.0,
                sigma=0):
    return utils.data.DataLoader(
        Loader(sample_rate, beta, df, noise_df, sigma, coinflip=coinflip),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True)


class Block(nn.Module):
    def __init__(self, n_input, n_output, kernel_size, stride, p):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(n_input)
        self.bn2 = nn.BatchNorm1d(n_output)

        self.dropout = nn.Dropout(p)

        self.conv1 = nn.Conv1d(n_input, n_output, kernel_size=kernel_size, padding=(kernel_size // 2), stride=stride)
        self.conv2 = nn.Conv1d(n_output, n_output, kernel_size=kernel_size, padding='same', stride=1)

        self.conv_res = nn.Conv1d(n_input, n_output, kernel_size=1, padding=2 if stride > 1 else 0, stride=stride)

    def forward(self, x):
        x_out = F.relu(self.bn1(x))
        # print(x_out.shape)
        out = (self.dropout(F.relu(self.bn2(self.conv1(x_out)))))
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        x_out = self.conv_res(x_out)
        # print(x_out.shape)
        return th.add(x_out, out)


class M5(nn.Module):
    def __init__(self, n_input=2, n_output=1, stride=16, n_channel=32, p=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        # self.conv1 = Block(n_input, n_channel, kernel_size=80, stride=stride, p=p)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = Block(n_channel, n_channel, kernel_size=3, stride=1, p=p)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = Block(n_channel, 2 * n_channel, kernel_size=3, stride=1, p=p)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        # x = th.logsumexp(x / x.shape[-1], dim=-1)

        x = self.dropout(x)
        x = self.fc1(x)
        return x.squeeze()


def infer(model, df, batch_size, sample_rate,):
    #model_train = model.train()
    res = []
    with th.no_grad():
        
        model.train(False)
        for x, y in make_loader(sample_rate, 0, df, batch_size, drop_last=False, shuffle=False, noise_df=None, sigma=0):
            res.append((model(x.cuda()) >= 0).cpu().numpy())
        #model.train(model_train)
        return np.concatenate(res)


def metrics(y, yh, pre=""):
    return {
        pre + "f1": f1_score(y, yh),
        pre + "bal_acc": balanced_accuracy_score(y, yh),
        pre + "acc": accuracy_score(y, yh),
        pre + "precision": precision_score(y, yh),
        pre + "recall": recall_score(y, yh),
    }

class Clf:
    def __init__(self, K, channel, lr, batch_size, sample_rate, epochs, negative_sample_rate, sigma, 
        beta, step_size, step_gamma, 
        optimizer,
        noise_df, coinflip, project,
        noise_df_=None):
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.epochs = epochs
        self.project = project
        self.negative_sample_rate = negative_sample_rate
        self.sample_rate = sample_rate
        self.sigma = sigma
        if optimizer == 'adam':
            self.optimizer = optim.Adam
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD
        elif optimizer == 'ada':
            self.optimizer = AdaBelief
        else:
            raise " optimizer not known"
        self.batch_size = batch_size
        self.beta = beta
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.noise_df = noise_df
        self.coinflip = coinflip
        self.noise_df_ = noise_df_
        self.lr = lr
        self.K = K
        self.channel = channel
    def fit(self, x, y=None, test_df=None):

   
        wandb.init(project=self.project, 
                entity="all-i-do-is-win",
                config={
                    'epochs': self.epochs,
                    'negative_sample_rate': self.negative_sample_rate,
                    'sample_rate': self.sample_rate,
                    'sigma': self.sigma,
                    'K': self.K,
                    'batch_size': self.batch_size,
                    'beta': self.beta,
                    'step_size': self.step_size,
                    'step_gamma': self.step_gamma,
                    'noise_df': self.noise_df,
                    'coinflip': self.coinflip,
                })
        self.model = M5(n_channel=self.channel).cuda()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        noise_opt = None
        noise = None
        lr_sched = optim.lr_scheduler.StepLR(opt, self.step_size, gamma=self.step_gamma)
        wandb.watch(self.model)
        train_df_gunshot = x[y]
        train_df_nothing = x[~y]


        self.model.train(True)
        for epoch in tqdm(range(self.epochs)):
            local_train_nothing = train_df_nothing.sample(
                min(len(train_df_nothing), 
                    len(train_df_gunshot) * self.negative_sample_rate))
            local_train_df = pd.concat([train_df_gunshot, 
                local_train_nothing]).reset_index()

            self.model.train(True)
            for x, y in make_loader(self.sample_rate, self.beta, 
                local_train_df, self.batch_size, 
                shuffle=True, drop_last=True, 
                noise_df=self.noise_df_ if self.noise_df else None,
                coinflip=self.coinflip):
                if self.K > 0 and (noise is None or noise.shape != x.shape):
                    noise = th.zeros_like(x, device='cuda')
                    noise_opt = optim.Adam([noise], lr=self.lr)
                y_cpu = y
                x, y = x.cuda(), y.cuda()

                for i in range(self.K):
                    self.model.zero_grad()
                    noise_opt.zero_grad()
                    logit = self.model(x + noise)
                    loss = -F.binary_cross_entropy_with_logits(logit, y)
                    loss.backward()
                    noise_opt.step()
                    with th.no_grad():
                        noise.clamp_(-1e-2, 1e-2)

                opt.zero_grad()
                if noise is not None:
                    x = x + noise
                
                logit = self.model(x)

                loss = F.binary_cross_entropy_with_logits(logit, y)
                loss.backward()
                wandb.log({"loss": loss.item()})
                opt.step()

                with th.no_grad():
                    p = (logit > 0).cpu().numpy()
                    wandb.log(metrics(y_cpu, p))
                lr_sched.step()
            if epoch % 10 == 0 and test_df is not None:
                res = test_df.copy()
                res.Label = self.pred(test_df)
                with tempfile.NamedTemporaryFile() as temp:
                    fn = f'{temp.name}-{epoch}.csv'
                    res.to_csv(fn, index=False)
                    wandb.save(fn)
                    
    def pred(self, x):
        return infer(self.model, x, self.batch_size, self.sample_rate)
def main(**args):
    df = pd.read_csv('data/participant_urbansound8k.csv')
    clf = Clf(project="real-run", noise_df_=df, **args)
    s = df.Label.isnull()
    train_df = pd.DataFrame(df[~s].reset_index())
    train_df.Label = (train_df.Label == True)
    test_df = df[s].reset_index()

    clf.fit(train_df, train_df.Label, test_df=test_df)

    res = test_df.copy()
    res.Label = clf.pred(test_df)
    with tempfile.NamedTemporaryFile() as temp:
        res.to_csv(temp.name+"-final.csv", index=False)
        wandb.save(temp.name+"-final.csv")
    return wandb
parser = argparse.ArgumentParser()
for i in ['batch_size', 'sample_rate', 'epochs', 'negative_sample_rate', 'channel', 'K']:
    parser.add_argument(f'--{i}', type=int)
for i in ['lr', 'beta', 'step_size', 'step_gamma', 'sigma', 'coinflip']:
    parser.add_argument(f'--{i}', type=float)
for i in ['noise_df']:
    parser.add_argument(f'--{i}', type=bool)
parser.add_argument("--optimizer")
if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
