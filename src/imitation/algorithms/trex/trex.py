import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


def make_fragments(traj, frag_len=25, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    T = len(traj.obs) - 1  # last obs has no action
    if T < frag_len: return []
    starts = rng.integers(0, T - frag_len + 1, size=max(1, T // frag_len))
    return [(traj.obs[s:s+frag_len], traj.rews) for s in starts]

def make_ranked_pairs(trajs, n_pairs=20000, frag_len=25, rng=None, tie_margin=0.0):
    rng = np.random.default_rng() if rng is None else rng
    # flatten all fragments
    frags = []
    for tr in trajs:
        frags += make_fragments(tr, frag_len, rng)
        print(len(frags))
    frags = [(o.astype(np.float32), r.astype(np.float32)) for o,r in frags]
    # sample pairs + label by true return
    pairs = []
    for _ in range(min(n_pairs, len(frags)**2)):
        i, j = rng.integers(0, len(frags), size=2)
        (o1, r1), (o2, r2) = frags[i], frags[j]
        g1, g2 = float(r1.sum()), float(r2.sum())
        if abs(g1-g2) <= tie_margin: continue
        y = 1.0 if g1 > g2 else 0.0
        pairs.append((o1, o2, y))
    return pairs

def collate_pad(batch):
    # pad sequences to same T (left pad=False → right pad with zeros)
    o1s, o2s, ys = zip(*batch)
    T1 = max(x.shape[0] for x in o1s); T2 = max(x.shape[0] for x in o2s)
    def pad(stack, T):
        D = stack[0].shape[1]
        out = torch.zeros(len(stack), T, D)
        for i,x in enumerate(stack):
            out[i,:x.shape[0]] = x
        return out
    return pad(o1s, T1), pad(o2s, T2), torch.stack(ys)

class PrefPairsDS(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        o1, o2, y = self.pairs[i]               # (T, obs_dim), (T, obs_dim), scalar
        return torch.from_numpy(o1), torch.from_numpy(o2), torch.tensor(y, dtype=torch.float32)

class RewardMLP(nn.Module):
    def __init__(self, obs_dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, 1)
        )
    def forward(self, x):            # x: (B, D)
        return self.net(x).squeeze(-1)


class TREX:

    #TODO: extract all parameters for training 
    num_epochs = 300
    ENV_NAME = 'CartPole-v1'

    # TODO:  different dataset object then tassos, currently just an array of trajectoryWithRew objects
    def train_fn(self, dataset, progress_fn):
        pairs = make_ranked_pairs(dataset)
        print(len(pairs))

        # Prepare Pytorch datasets for training/validation
        pairs_dataset = PrefPairsDS(pairs)
        val_fraction = 0.2
        val_size = max(1, int(len(pairs_dataset) * val_fraction))
        train_size = len(pairs_dataset) - val_size
        if train_size <= 0:
            raise ValueError('Not enough preference pairs to create a training split.')
        generator = torch.Generator().manual_seed(0)
        train_dataset, val_dataset = random_split(pairs_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_pad)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_pad)

        #TODO: still relies on Gym Environment, instead of our own wrapper
        obs_dim = gym.make(self.ENV_NAME).observation_space.shape[0]

        reward_net = RewardMLP(obs_dim)
        opt = torch.optim.Adam(reward_net.parameters(), lr=3e-4)

        #TODO: cpu is still hardcoded
        device = torch.device("cpu"); reward_net.to(device)


        def pred_frag_return(obs_seq):       # (B, T, D) -> (B,)
            B,T,D = obs_seq.shape
            r = reward_net(obs_seq.reshape(B*T, D)).view(B,T).sum(dim=1)
            return r

        def bt_loss(r1, r2, y):              # y in {0,1}
            return F.binary_cross_entropy_with_logits(r1 - r2, y)
        
        def run_epoch(data_loader, train=True):
            reward_net.train(mode=train)
            total_loss, n_batches = 0.0, 0
            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                for o1, o2, y in data_loader:
                    o1, o2, y = o1.to(device), o2.to(device), y.to(device)
                    T = min(o1.shape[1], o2.shape[1])
                    o1, o2 = o1[:, :T], o2[:, :T]
                    r1, r2 = pred_frag_return(o1), pred_frag_return(o2)
                    loss = bt_loss(r1, r2, y)
                    if train:
                        opt.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(reward_net.parameters(), 5.0)
                        opt.step()
                    total_loss += loss.item()
                    n_batches += 1
            return total_loss / max(1, n_batches)
        
        
        for epoch in range(1, self.num_epochs + 1):
            train_loss = run_epoch(train_loader, train=True)
            val_loss = run_epoch(val_loader, train=False)
            metrics = {
                "epoch": epoch,
                "num_epochs": self.num_epochs,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            progress_fn(metrics)