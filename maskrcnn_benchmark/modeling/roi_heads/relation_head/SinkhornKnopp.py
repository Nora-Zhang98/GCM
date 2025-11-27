class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05, imb_factor=1):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.imb_factor = imb_factor

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)  # remove inf
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]

        # obtain permutation/order from the marginals
        marginals_argsort = torch.argsort(Q.sum(1))
        marginals_argsort = marginals_argsort.detach()
        r = []
        for i in range(Q.shape[0]):
            r.append((1 / self.imb_factor) ** (i / (Q.shape[0] - 1.0)))

        r = np.array(r)
        r = r * (Q.shape[1] / Q.shape[0])
        r = torch.from_numpy(r).cuda(non_blocking=True)
        r[marginals_argsort] = torch.sort(r)[0]  # Sort/permute based on the data order
        r = torch.clamp(r, min=1)  # Clamp the min to have a balance distribution for the tail classes
        r /= r.sum()  # Scaling to make it prob
        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits, prior_P=None):
        # get assignments
        # import pdb; pdb.set_trace()
        B, K = logits.shape
        if prior_P is not None:
            denominator = self.epsilon + self.epsilon2
            prior_withreg = - torch.log(prior_P / B) * self.epsilon2
            q = (logits + prior_withreg) / denominator
        else:
            q = logits / self.epsilon

        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q)
