import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torchaudio
import torchaudio.functional as Fa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, precision_score, recall_score
import wandb
from tqdm.auto import tqdm

def replicate(x, xn):
    while xn.shape[1] <= x.shape[1]:
        xn = th.cat([xn, xn], dim=1)
    if xn.shape[1] > x.shape[1]:
        xn = xn[:, :x.shape[1]]
    return xn


class Loader:
    def __init__(self, sample_rate, beta, df, noise_df=None, noise_elem=None, sigma=0):
        self.df = df
        self.noise_df = noise_df
        self.sigma = sigma
        self.beta = beta
        self.sample_rate = sample_rate
        self.noise_elem = self.f(noise_elem)[0] if noise_elem is not None else None

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

        if self.noise_df is not None:
            xn, _ = self.f(self.noise_df.sample(n=1).iloc[0])

            xn = replicate(x, xn)
            x = x + self.beta * xn
        if self.noise_elem is not None:
            xn = replicate(x, self.noise_elem)
            snr = 10**(self.beta/20) * x.norm(2) / xn.norm(2)
            x = x + snr * xn
        # if self.sigma > 0:
        #     x = x + self.randn(x.shape) * self.sigma
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
                noise_elem=None,
                num_workers=8,
                sigma=0):
    return utils.data.DataLoader(
        Loader(sample_rate, beta, df, noise_df, noise_elem, sigma),
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
        # x = F.avg_pool1d(x, x.shape[-1])
        # x = x.permute(0, 2, 1)
        x = th.logsumexp(x / x.shape[-1], dim=-1)

        x = self.dropout(x)
        x = self.fc1(x)
        return x.squeeze()


def infer(model, df, batch_size, sample_rate,):
    #model_train = model.train()
    res = []
    with th.no_grad():
        
        model.train(False)
        for x, y in make_loader(sample_rate, 0, df, batch_size, drop_last=False, shuffle=False, noise_df=None, noise_elem=None, sigma=0):
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


def main(batch_size, sample_rate, epochs, negative_sample_rate,
            sigma, beta, val_rate, step_size, step_gamma, train_noise_elem, noise_df):
    wandb.login()
    wandb.init(project="real-run", entity="all-i-do-is-win",
               config={
                   'epochs': epochs,
                   'negative_sample_rate': negative_sample_rate,
                   'sample_rate': sample_rate,
                   'sigma': sigma,
                   'val_rate': val_rate,
                   'batch_size': batch_size,
                   'beta': beta,
               })

    df = pd.read_csv('data/participant_urbansound8k.csv')
    s = df.Label.isnull()
    train_val_df = pd.DataFrame(df[~s].reset_index())
    train_val_df.Label = (train_val_df.Label == True)
    test_df = df[s].reset_index()

    train_df, val_df = train_test_split(train_val_df, train_size=1 - val_rate) if val_rate != 0 else train_val_df, []
    train_df_gunshot = train_df[train_df['Label']]
    train_df_nothing = train_df[~train_df['Label']]

    model = M5().cuda()
    opt = optim.Adam(model.parameters())

    lr_sched = optim.lr_scheduler.StepLR(opt, step_size, gamma=step_gamma)
    wandb.watch(model)

    model.train(True)
    for epoch in tqdm(range(epochs)):
        local_train_nothing = train_df_nothing.sample(
            min(len(train_df_nothing), len(train_df_gunshot) * negative_sample_rate))
        local_train_df = pd.concat([train_df_gunshot, local_train_nothing]).reset_index()

        model.train(True)
        for x, y in make_loader(sample_rate, beta, local_train_df, batch_size, 
                shuffle=True, drop_last=True, 
                noise_elem=df.iloc[train_noise_elem] if train_noise_elem is not None else None,
                noise_df=df if noise_df else None):

            opt.zero_grad()
            y_cpu = y
            x, y = x.cuda(), y.cuda()

            logit = model(x)

            loss = F.binary_cross_entropy_with_logits(logit, y)
            loss.backward()
            wandb.log({"loss": loss.item()})
            opt.step()

            with th.no_grad():
                p = (logit > 0).cpu().numpy()
                wandb.log(metrics(y_cpu, p))
        if len(val_df) > 0:
            with th.no_grad():
                vy = val_df.Label == 1
                vyh = infer(model, val_df)
                wandb.log(metrics(vy, vyh, pre="val_"))
        lr_sched.step()
    res = test_df.copy()
    res.Label = infer(model, test_df, batch_size*4, sample_rate)
    res.to_csv('submit.csv', index=False)
    wandb.save('submit.csv')


main(batch_size=64, sample_rate=10000, epochs=200, negative_sample_rate=2,
     sigma=0.00, beta=3.0, val_rate=0.0, step_size=50, step_gamma=0.5, train_noise_elem=None, noise_df=True)
