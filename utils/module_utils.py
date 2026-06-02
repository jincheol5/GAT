import torch

class ModuleUtils:
    @staticmethod
    def group_softmax(
            score:torch.Tensor,
            index:torch.Tensor,
            num_nodes:int
        ):
        """
        target node별 softmax.
        numerical stability를 위해 softmax 계산 시 각 target node의 가장 큰 score 값은 빼고 계산

        score: [E,n_head]
        index: [E,]  # target node id
        """
        E,n_head=score.size()

        # numerical stability를 위해 target node별 max 저장 공간 생성, 각 node마다 head별 최대 score를 저장
        max_score=torch.full(
            (num_nodes,n_head),
            -float("inf"),
            device=score.device,
            dtype=score.dtype
        ) # [N,n_head]

        # 각 node마다 head별 최대 score를 계산
        max_score.scatter_reduce_(
            dim=0,
            index=index.unsqueeze(-1).expand(E,n_head), # index.shape == src.shape이여야 함, [E,] -> [E,n_head]
            src=score,
            reduce="amax",
            include_self=True
        ) # [N,n_head]

        # softmax의 분자 계산
        score_exp=torch.exp(score-max_score[index]) # [E,n_head]

        # softmax의 분모 계산
        denom=torch.zeros(
            num_nodes,
            n_head,
            device=score.device,
            dtype=score.dtype
        )
        denom.index_add_(dim=0,index=index,src=score_exp)
        softmax_score=score_exp/(denom[index]+1e-16)
        return softmax_score # [E,n_head]