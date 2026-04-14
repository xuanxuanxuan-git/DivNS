import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from time import time
from collections import defaultdict
# from dppy.finite_dpps import FiniteDPP


class MF(nn.Module):
    def __init__(self, data_config, args):
        super(MF, self).__init__()
        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]
        self.emb_size = args.dim
        self.l2 = args.l2
        # self.offset = min(stats["all_item"])
        self.ns = args.ns
        self.K = args.K
        self.n_negs = args.n_negs
        self.dataset=args.dataset
        self.pool = args.pool
        self.topk = args.topk
        self.device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
        ### --- AHNS
        self.alpha = args.alpha
        self.beta = args.beta
        self.p = args.p

        ### --- DENS
        self.warmup = args.warmup
        # gating
        self.gamma = args.gamma 
        self.user_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.item_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)

        self.pos_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.neg_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)

        """ Init the weight of user and item embedding """
        self.embedding_dict = self.init_weight()

        ### --- recycle historical negatives
        self.cached_negs_idx = defaultdict(list) # {user_id: [neg_item_id]}
        self.latest_used_negs_idx = defaultdict(list)
        self.recycle_negs_idx = defaultdict(list)

    def init_weight(self):
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            "user_emb": nn.Parameter(initializer(torch.empty(self.n_users, self.emb_size))),
            "item_emb": nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size)))
        })

        return embedding_dict

    def create_bpr_loss(self, user_embed, pos_item_embed, neg_item_embed):
        batch_size = user_embed.shape[0]

        u_e = user_embed
        pos_e = pos_item_embed
        neg_e = neg_item_embed

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))

        # TODO: BCE loss, SSM loss
        # bce_loss = torch.log(torch.exp(neg_scores)/(1+torch.exp(neg_scores))) + torch.log(1+torch.exp(-pos_scores)) # not sure if we should take the average
        # ssm_loss = torch.mean(torch.log(torch.exp(pos_scores)/ torch.exp(pos_scores)+torch.exp(neg_scores).sum(dim=1)))

        # cul regularizer
        regularize = (torch.norm(user_embed[:, :]) ** 2
                       + torch.norm(pos_item_embed[:, :]) ** 2
                       + torch.norm(neg_item_embed[:, :]) ** 2) / 2  # take hop=0
        # TODO: for ncf etc., reg term should also add FC and W weight (maybe not)
        emb_loss = self.l2 * regularize / batch_size
        return mf_loss + emb_loss, pos_scores.detach(), neg_scores.detach()

    # inference function
    def rating(self, u_g_embeddings=None, pos_i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def generate(self, split=True):
        user_emb, item_emb = self.embedding_dict['user_emb'], self.embedding_dict['item_emb']
        if split:
            return user_emb, item_emb
        else:
            return torch.cat([user_emb, item_emb], dim=0)
    
    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]
    
    def forward(self, cur_epoch, batch):
        user = batch['users']      # [batch_size]
        pos_item = batch['pos_items']       # [batch_size]
        neg_item = batch['neg_items']
        current_batch_selections = torch.stack((user, pos_item), dim=1)  

        user_embed = self.embedding_dict['user_emb'][user]
        pos_item_embed = self.embedding_dict['item_emb'][pos_item]
        # neg_item_embs = self.embedding_dict['item_emb'][neg_item]
        all_user_embed = self.embedding_dict["user_emb"]
        all_item_embed = self.embedding_dict["item_emb"]

        neg_item_embs = []

        if self.ns in ["rns", "pns"]:
            neg_item_embs = all_item_embed[neg_item[:, :self.K]]
            current_batch_selections = torch.cat((current_batch_selections, neg_item[:, :self.K]), dim=1)
        
        elif self.ns == "dns":

            for k in range(self.K):
                neg_item_emb, neg_item_id = self.dynamic_negative_sampling(all_user_embed, all_item_embed, user, neg_item[:, k*self.n_negs:(k+1)*self.n_negs])
                neg_item_embs.append(neg_item_emb)
            neg_item_embs = torch.stack(neg_item_embs, dim=1) # [batch_size, K, channel]

            # synthesise negative samples
            neg_item_embs = self.synthesise_historical_negatives(all_item_embed, cur_epoch, user, neg_item_embs)

        elif self.ns == "dns_mn":
            for k in range(self.K):
                neg_item_embs.append(self.dynamic_mn_negative_sampling(all_user_embed, all_item_embed, user, neg_item[:, k*self.n_negs:(k+1)*self.n_negs]))
            neg_item_embs = torch.stack(neg_item_embs, dim=1)
        
        elif self.ns == "dens":
            for k in range(self.K):
                neg_item_embs.append(self.dise_negative_sampling(cur_epoch, all_user_embed, all_item_embed, user, neg_item[:, k*self.n_negs:(k+1)*self.n_negs], pos_item))
            neg_item_embs = torch.stack(neg_item_embs, dim=1)
        
        elif self.ns == "ahns":
            for k in range(self.K):
                neg_item_embs.append(self.adaptive_negative_sampling(all_user_embed, all_item_embed, user, neg_item[:, k*self.n_negs:(k+1)*self.n_negs], pos_item))
            neg_item_embs = torch.stack(neg_item_embs, dim=1)
            
        return self.create_bpr_loss(user_embed, pos_item_embed, neg_item_embs)


    def synthesise_historical_negatives(self, all_item_embs, cur_epoch, user_id, neg_item_embs):
        # first epoch, cache is empty, do not synthesise
        if cur_epoch == 0:
            return neg_item_embs
        
        if self.dataset=="ml-1m":
            cached_negs_embs = neg_item_embs    # comment out if not testing
        else:
            cached_neg_indices = torch.stack([self.recycle_negs_idx[u.item()][0] for u in user_id])

            for u in user_id:
                self.recycle_negs_idx[u.item()] = self.recycle_negs_idx[u.item()][1:]
            cached_negs_embs = all_item_embs[cached_neg_indices]  # [batch_size, 1, emb_dim]          # comment back
        # cached_negs_embs = all_item_embs[cached_neg_indices].unsqueeze(1)  # [batch_size, 1, emb_dim]

        # Weighted combination
        weight_1 = 0.8
        syn_neg_embs = weight_1 * neg_item_embs + (1-weight_1) * cached_negs_embs

        # Compute the mean of cached and new negative embeddings
        # syn_neg_embs = torch.sum(torch.stack([cached_negs_embs, neg_item_embs], dim=-1), dim=-1)  # pool along the third dimension

        return syn_neg_embs     # [batch_size, 1, dim]
    
    
    def choose_from_cache_non_greedy(self, neg_item_ids:list, cached_negs_ids:list):
        num_inter = len(neg_item_ids)

        all_item_embed = self.embedding_dict["item_emb"]
        unique_cached_negs_ids = torch.unique(torch.tensor(cached_negs_ids)) 
         
        cached_item_embed = all_item_embed[unique_cached_negs_ids]
        used_item_embed = all_item_embed[torch.tensor(neg_item_ids)] 
        
        # start normal sampling
        min_distances = min_cosine_distance(cached_item_embed, used_item_embed)
        top_k_indices = torch.topk(min_distances, k=min(num_inter,len(unique_cached_negs_ids)), largest=False).indices
        
        # Select the corresponding items
        selected_item_ids = unique_cached_negs_ids[top_k_indices]
    
        return selected_item_ids 


    def choose_from_cache_greedy(self, neg_item_ids: list, cached_negs_ids: list):
        num_inter = len(neg_item_ids)

        all_item_embed = self.embedding_dict["item_emb"]
        unique_cached_negs_ids = torch.unique(torch.tensor(cached_negs_ids))
        
        cached_item_embed = all_item_embed[unique_cached_negs_ids]
        used_item_embed = all_item_embed[torch.tensor(neg_item_ids)] 
        
        # Start greedy sampling
        selected_item_indices = []  # Store indices relative to cached_item_embed
        selected_item_embs = []    # Store selected embeddings
        
        # Make a copy of the cached embeddings to modify during selection
        remaining_cached_emb = cached_item_embed.clone()

        for _ in range(num_inter):
            # Calculate the minimum cosine distance between each candidate and the set of used items
            min_distances = min_cosine_distance(remaining_cached_emb, used_item_embed)
            
            # Select the candidate with the maximum minimum distance (maximize diversity)
            best_candidate_idx = torch.argmax(min_distances).item()
            
            # Get the selected item's embedding
            selected_item_emb = remaining_cached_emb[best_candidate_idx]
            selected_item_embs.append(selected_item_emb)
            selected_item_indices.append(best_candidate_idx)
            
            # Remove the selected item from the candidate set (so we don't select it again)
            mask = torch.ones(len(remaining_cached_emb), dtype=torch.bool)
            mask[best_candidate_idx] = False
            remaining_cached_emb = remaining_cached_emb[mask]
            
            # Add the selected item to the used_item_embed
            used_item_embed = torch.cat([used_item_embed, selected_item_emb.unsqueeze(0)])

        # Convert the selected indices back to the original item IDs
        selected_in_unique = torch.tensor(selected_item_indices)
        selected_item_ids = unique_cached_negs_ids[selected_in_unique]
        
        return selected_item_ids


    def recycle_cached_samples(self):
        """At the end of each epoch, we recycle samples and determine which to reuse."""
        # for every single user
        for user_id, neg_item_ids in self.latest_used_negs_idx.items():
            cached_negs_ids = self.cached_negs_idx[user_id]
            recycled_negs = self.choose_from_cache_non_greedy(neg_item_ids, cached_negs_ids)
            self.recycle_negs_idx[user_id] = recycled_negs
        
        return True


    def dynamic_negative_sampling(self, user_emb, item_emb, user, neg_candidates):
        # select the highest ranked negative item
        s_e = user_emb[user]  # [batch_size, channel]
        n_e = item_emb[neg_candidates]  # [batch_size, n_negs, channel]

        """dynamic negative sampling"""
        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs]      

        num_cache = 5
        top_k_indices = torch.topk(scores, k=(num_cache+1), dim=-1)[1]      # [batch_size, k_indices] top k indices
        top_1_indices = top_k_indices[:, 0]                       # [batch_size]
        top_1_actual_idx = torch.gather(neg_candidates, dim=1, index=top_1_indices.unsqueeze(-1)).squeeze()

        top_unused_indices = top_k_indices[:, 1:]
        cached_neg_indices = torch.gather(neg_candidates, dim=1, index=top_unused_indices)      # [batch_size, keep_num=10]: retain top hard negatives
        
        for u, cached in zip(user, cached_neg_indices):
            self.cached_negs_idx[u.item()].extend(cached.tolist())
        
        for u, used in zip(user, top_1_indices):
            self.latest_used_negs_idx[u.item()].append(used.item())

        return item_emb[top_1_actual_idx], top_1_actual_idx
    
    
    def dynamic_mn_negative_sampling(self, user_emb, item_emb, user, neg_candidates):
        # N is the pool size
        # M is the top-k and used to control the probability distribution for high-ranked negative items.
        # For negative samples with top-M predicted scores, each has a prob of 1/M.
        # M = 5/6 for metric@20. N=200
        batch_size = user.shape[0]
        s_e = user_emb[user]  # [batch_size, channel]
        n_e = item_emb[neg_candidates]  # [batch_size, n_negs, channel]

        """dynamic negative sampling with pre-fixed M"""
        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs]
        indices = torch.topk(scores, self.topk, dim=1)[1].detach()  # [batch_size, topk]

        probs = torch.tensor([1.0 / (i + 1) for i in range(self.topk)])
        probs = probs / probs.sum()
        probs_batch = probs.unsqueeze(0).repeat(batch_size, 1) # [batch_size, topk]

        distribution = torch.distributions.Categorical(probs_batch)
        selected_indices = distribution.sample().to(self.device)    # [batch_size]

        # selected_indices = torch.randint(0, self.topk, (batch_size,)) # [batch_size]
        result_indices = torch.gather(indices, dim=1, index=selected_indices.unsqueeze(1)).squeeze() # [batch_size]
        
        neg_item = torch.gather(neg_candidates, dim=1, index=result_indices.unsqueeze(-1)).squeeze()
        
        return item_emb[neg_item]

    def dise_negative_sampling(self, cur_epoch, user_emb, item_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_emb[user], item_emb[pos_item]  # [batch_size, channel]
        n_e = item_emb[neg_candidates]  # [batch_size, n_negs, channel]
        
        gate_p = torch.sigmoid(self.item_gate(p_e) + self.user_gate(s_e))
        gated_p_e = p_e * gate_p    # [batch_size, channel]

        gate_n = torch.sigmoid(self.neg_gate(n_e) + self.pos_gate(gated_p_e).unsqueeze(1))
        gated_n_e = n_e * gate_n    # [batch_size, n_negs, channel]

        n_e_sel = (1 - min(1, cur_epoch / self.warmup)) * n_e - gated_n_e    # [batch_size, n_negs, channel]
        # n_e_sel = (1 - max(0, 1 - (cur_epoch / self.warmup))) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]
        # n_e_sel = (1 - self.alpha) * n_e - gated_n_e    # [batch_size, n_negs, n_hops+1, channel]

        """dynamic negative sampling"""
        scores = (s_e.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs]
        indices = torch.max(scores, dim=1)[1].detach()
         
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()

        # neg_items_emb_ = n_e
        # # .permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # # [batch_size, n_hops+1, channel]
        
        # neg_items = neg_items_emb_[[[i] for i in range(batch_size)],
        #     #    range(neg_items_emb_.shape[1]), 
        #        indices, :]

        return item_emb[neg_item]
    
    def adaptive_negative_sampling(self, user_emb, item_emb, user, neg_candidates, pos_item):

        s_e, p_e = user_emb[user], item_emb[pos_item]  # [batch_size, channel]
        n_e = item_emb[neg_candidates]  # [batch_size, n_negs, channel]
   
        p_scores = (s_e*p_e).sum(dim=-1).unsqueeze(dim=1)   # [batch_size, 1]
        n_scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs]

        scores = torch.abs(n_scores - self.beta * (p_scores + self.alpha).pow(self.p + 1))

        """adaptive negative sampling"""
        indices = torch.min(scores, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze()
        
        return item_emb[neg_item]
    
    def dpp_sampling(self, neg_item_ids, cached_negs_ids):
        num_inter = len(neg_item_ids)

        cached_negs_ids = torch.stack(cached_negs_ids)
        cached_negs_ids = cached_negs_ids.view(-1)
        perm = torch.randperm(len(cached_negs_ids), device=cached_negs_ids.device)
        sampled_negs_ids = cached_negs_ids[perm[:num_inter]].tolist()
        
        # random.shuffle(cached_negs_ids)
        # sampled_negs_ids = cached_negs_ids[:num_inter]

        return sampled_negs_ids

        # # select the set that is diverse from the chosen negs
        # all_item_embed = self.embedding_dict["item_emb"]
        # unique_cached_negs_ids = torch.unique(torch.cat(cached_negs_ids)) 
        # cached_item_embed = all_item_embed[unique_cached_negs_ids]
        # used_item_embed = all_item_embed[neg_item_ids]
        
        # # S = cosine_similarity_matrix(cached_item_embed)
        # S = rbf_kernel(cached_item_embed)
        
        # # Compute average similarity to target set for each candidate
        # similarity_to_target = torch.mean(
        #     torch.mm(cached_item_embed, used_item_embed.T) / (torch.norm(cached_item_embed, dim=1, keepdim=True) * torch.norm(used_item_embed, dim=1) + 1e-8),
        #     dim=1
        # )

        # # Define re-weighting factor
        # alpha = 1.0  # Controls strength of the penalty
        # weights = 1 / (1 + alpha * similarity_to_target)

        # # Re-weight the kernel
        # K = torch.outer(weights, weights) * S

        # # Sample using DPP
        # dpp = FiniteDPP('likelihood', L=K.detach().numpy())
        # dpp.sample_exact(k=num_inter)
        # # Get selected indices
        # sampled_negs_ids = dpp.list_of_samples[-1]

        # if not sampled_negs_ids:
        #     # print(unique_cached_negs_ids)
        #     return unique_cached_negs_ids[:num_inter]

        # return unique_cached_negs_ids[sampled_negs_ids]

# Compute cosine similarity between candidate items
def cosine_similarity_matrix(E):
    norm = torch.norm(E, p=2, dim=1, keepdim=True)
    E_normalized = E / (norm + 1e-6)
    return torch.matmul(E_normalized, E_normalized.T)

def rbf_kernel(E, gamma=1.0):
    pairwise_sq_dists = torch.cdist(E, E, p=2) ** 2
    return torch.exp(-gamma * pairwise_sq_dists)


def min_cosine_distance(candidates, target_set):
    # Normalize embeddings for cosine similarity calculation
    candidates_norm = F.normalize(candidates, p=2, dim=1)
    target_set_norm = F.normalize(target_set, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.matmul(candidates_norm, target_set_norm.T)
    
    # Get the minimum similarity for each candidate with the target set
    min_similarities = cosine_sim.min(dim=1).values
    return min_similarities