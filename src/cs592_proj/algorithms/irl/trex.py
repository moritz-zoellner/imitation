import numpy as np
from tqdm import trange

from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from cs592_proj import algorithms


def make_fragments(traj, frag_len=25, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    T = len(traj["observations"]) - 1  # last obs has no action
    if T < frag_len: return []
    starts = rng.integers(0, T - frag_len + 1, size=max(1, T // frag_len))
    observations = np.array(traj["observations"])
    rewards = np.array(traj["reward"])
    return [(observations[s:s+frag_len], rewards) for s in starts]

def make_ranked_pairs(trajs, n_pairs=20000, frag_len=25, rng=None, tie_margin=0.0):
    rng = np.random.default_rng() if rng is None else rng
    # flatten all fragments
    frags = []
    for tr in trajs:
        print(tr)
        frags += make_fragments(tr, frag_len, rng)
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
        return self.net(x)
    

class LearnedRewardEnv(gym.Wrapper):
    """Wraps env to replace rewards with learned reward network outputs."""
    def __init__(self, env, reward_model, device):
        super().__init__(env)
        self.reward_model = reward_model
        self.device = device

    def step(self, action):
        obs, true_reward, terminated, truncated, info = self.env.step(action)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            learned_reward = self.reward_model(obs_tensor).item()
        info = dict(info)
        info.setdefault('true_reward', true_reward)
        return obs, learned_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class TREX:

    #TODO: extract all parameters for training 
    num_epochs = 30
    learned_reward_timesteps = 200000
    episode_length: int = 1000
    action_repeat: int = 1
    policy_learning_algo: str = "DQN"

    # TODO:  different dataset object then tassos, currently just an array of trajectoryWithRew objects
    def train_fn(self, *, run_config, dataset, progress_fn, **_):
        pairs = make_ranked_pairs(dataset)

        # -------- Prepare Pytorch datasets for training/validation -------------
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

        eval_env = dataset.get_eval_env(episode_length=self.episode_length, action_repeat=self.action_repeat)
        obs_dim = eval_env.obs_size

        reward_net = RewardMLP(obs_dim)
        opt = torch.optim.Adam(reward_net.parameters(), lr=3e-4)

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")

        #--------------- Training the reward net -------------------
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
        
        for epoch in trange(1, self.num_epochs + 1):
            train_loss = run_epoch(train_loader, train=True)
            val_loss = run_epoch(val_loader, train=False)
            #TODO definitely different metrics 
            metrics = {
                "epoch": epoch,
                "num_epochs": self.num_epochs,
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            print(metrics) 

        #-------------- Building a policy from trained reward net ------------
        
        print("start_learning")

        reward_net.eval()
        env = dataset.get_env()
        env.override_training_reward_fn(reward_net)

        if hasattr(algorithms, self.policy_learning_algo):
            AlgoClass = getattr(algorithms, self.policy_learning_algo)
        else:
            raise NotImplementedError(f"Algorithm {self.policy_learning_algo} not found")

        algo = AlgoClass()

        make_policy, params, metrics = algo.train_fn(
            run_config=run_config,
            env=env,
            progress_fn=progress_fn
        )

        return make_policy, params, metrics
