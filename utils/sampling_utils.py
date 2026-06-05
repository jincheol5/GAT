import numpy as np
import torch

class SamplingUtils:
    """
    edge_index에 self-loop 포함해야 함.
    """
    @staticmethod
    def convert_edge_index_to_idx(
            num_nodes:int,
            edge_index:torch.Tensor
        ):
        """
        Input:
            num_nodes: int
            edge_index: [2,E]
        Output:
            idx: [E,]
            population: int
        """
        src,tar=edge_index

        # edge(src,tar)을 edge 번호로 mapping: edge_id = src x N + tar
        idx=src*num_nodes+tar # [E,]

        # population는 가능한 edge 수: N x N
        population=num_nodes*num_nodes
        return idx,population

    @staticmethod
    def negative_sampling(
            num_nodes:int,
            edge_index:torch.Tensor,
            num_neg_edge:int=None
        ):
        """
        가능한 모든 edge를 번호로 매긴 뒤, 실제 edge 번호를 제외하고 랜덤 번호를 뽑아 negative edge 생성
        
        negative edge수는 positive edge 수와 최대한 동일하도록 함
            -positive edge 수가 가능한 edge 수의 절반 이상일 경우 비율 조정
        
        edge_id = src x N + tar

        Input:
            num_nodes: int
            edge_index: [2,E]
            num_neg_edge: int
        Output:

        """
        ### 1. Existing edge ids
        idx,population=SamplingUtils.convert_edge_index_to_idx(
            num_nodes=num_nodes,
            edge_index=edge_index
        )

        # graph가 꽉 찬 경우
        if idx.numel()>=population:
            return edge_index.new_empty((2,0))
        

        ### 2. Negative edge 수 계산
        # negative edge수는 positive edge 수와 동일
        if num_neg_edge is None:
            num_neg_edge=edge_index.size(1)

        # Maximum available negatives
        max_neg_edge=population-idx.numel()

        # Cannot sample more than available
        num_neg_edge=min(
            num_neg_edge,
            max_neg_edge
        )

        ### 3. Negative sampling probability
        # 범위 안에서 random 생성 시 필요한 negative 개수를 얻기 위해 미리 넉넉하게 뽑을 랜덤 번호 개수 필요
        prob=1.-idx.numel()/population  # Probability to sample a negative.
        sample_size=int(1.1*num_neg_edge/prob)  # Oversampling size


        ### 4. Main sampling loop
        # PyG에서는 3번만 반복하며 negative sampling
        idx_cpu=idx.cpu().numpy() # CPU for np.isin
        neg_idx=None
        for _ in range(3):
            # Random edge ids
            rnd=torch.randint(
                low=0,
                high=population,
                size=(sample_size,),
                device='cpu'
            )

            # Remove true edges
            mask=np.isin(rnd.numpy(),idx_cpu)

            # Remove already sampled negatives
            if neg_idx is not None:
                mask |= np.isin(
                    rnd.numpy(),
                    neg_idx.cpu().numpy()
                )
            mask=torch.from_numpy(mask)
            rnd=rnd[~mask]
            rnd=torch.unique(rnd) # 중복 edge 제거

            # Accumulate negatives
            if neg_idx is None:
                neg_idx=rnd
            else:
                neg_idx=torch.cat(
                    [neg_idx,rnd],
                    dim=0
                )
            
            # Enough negatives collected
            if neg_idx.numel()>=num_neg_edge:
                neg_idx=neg_idx[:num_neg_edge]
                break
        
        ### 5. edge_id -> (src, tar)
        neg_src=neg_idx//num_nodes
        neg_tar=neg_idx%num_nodes
        neg_edge_index=torch.stack(
            [neg_src,neg_tar],
            dim=0
        ).to(edge_index.device)
        return neg_edge_index

    @staticmethod
    def neighbor_sampling(
            edge_index:torch.Tensor,
            pos_edge_index:torch.Tensor,
            neg_edge_index:torch.Tensor,
            n_hop:int=1
        ):
        """
        Input:
            edge_index
            pos_edge_index
            neg_edge_index
            n_hop
        Output:
            sub_edge_index
        """
        src,tar=edge_index
        edge_label_index=torch.cat(
            [pos_edge_index,neg_edge_index],
            dim=1
        )

        seed_nodes=torch.cat([
            edge_label_index[0],
            edge_label_index[1]
        ], dim=0)
        seed_nodes=seed_nodes.unique()

        sampled_edge_ids=[]
        for _ in range(n_hop):
            # seed node들의 이웃 노드들과 들어오는 edge들을 구함
            mask=torch.isin(tar,seed_nodes) # [E,]
            hop_edge_ids=mask.nonzero(as_tuple=False).view(-1) # True인 위치 
            sampled_edge_ids.append(hop_edge_ids)

            # next seed node들을 현재 seed node들의 이웃 노드들로 지정
            seed_nodes=src[hop_edge_ids].unique()
        
        # compute sub_edge_index
        sampled_edge_ids=torch.unique(torch.cat(sampled_edge_ids,dim=0)) # unique로 자동 정렬
        sub_edge_index=edge_index[:,sampled_edge_ids]
        return sub_edge_index