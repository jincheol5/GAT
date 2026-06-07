import numpy as np
import torch
from typing_extensions import Literal

class SamplingUtils:
    """
    edge_index에 self-loop 포함해야 함.
    """
    @staticmethod
    def remove_self_loop_in_edge_index(
            edge_index:torch.Tensor
        ):
        """
        Input:
            edge_index:
        Return:
            updated_edge_index:
        """
        src,tar=edge_index
        mask=src!=tar
        updated_edge_index=edge_index[:,mask]
        return updated_edge_index
    
    @staticmethod
    def add_self_loop_to_edge_index(
            node:torch.Tensor,
            edge_index:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            edge_index: [2,E]
        Return:
            updated_edge_index
        """
        self_loop_edge_index=torch.stack(
            [node,node],
            dim=0
        )
        updated_edge_index=torch.cat(
            [edge_index,self_loop_edge_index],
            dim=1
        )
        return updated_edge_index

    @staticmethod
    def convert_edge_index_to_idx(
            num_node:int,
            edge_index:torch.Tensor
        ):
        """
        Input:
            num_node: int
            edge_index: [2,E]
        Output:
            idx: [E,]
            population: int
        """
        src,tar=edge_index

        # edge(src,tar)을 edge 번호로 mapping: edge_id = src x N + tar
        idx=src*num_node+tar # [E,]

        # population는 가능한 edge 수: N x N
        population=num_node*num_node
        return idx,population

    @staticmethod
    def negative_sampling(
            num_node:int,
            edge_index:torch.Tensor,
            num_neg_edge:int=1
        ):
        """
        전체 edge_index에 존재하지 않는 edge 중에서
        batch_edge_index 개수만큼 negative edge를 생성.

        self-loop는 negative edge로 생성되지 않도록 코드 작성.

        test neg edge 먼저 생성 후, train neg edge 생성 시 전체 edge_index에 test_neg_edge 같이 추가해서 sampling  
        -> 학습용 neg edge를 평가에도 사용하는 경우 피하기 위해서 수행
        
        edge_id = src * N + tar

        Input:
            num_node: int
            edge_index: [2, E], 전체 edge index
            num_neg_edge: int, 생성할 negative edge 수, batch size만큼 생성

        Output:
            neg_edge_index: [2, num_neg_edge]
        """
        ### 1. Existing edge ids
        idx,population=SamplingUtils.convert_edge_index_to_idx(
            num_node=num_node,
            edge_index=edge_index
        )

        # graph가 꽉 찬 경우
        if idx.numel()>=population:
            return edge_index.new_empty((2,0))

        ### 2. Negative edge 수 계산
        # 기본값: batch positive edge 수만큼 negative 생성
        # 가능한 최대 negative 수
        max_neg_edge=population-idx.numel()-num_node # 전체 가능 edge 수 - 현재 존재하는 edge 수 - self-loop수

        # 가능한 negative 수보다 많이 뽑을 수 없음
        num_neg_edge=min(num_neg_edge,max_neg_edge)

        ### 3. Negative sampling probability
        prob=1.-idx.numel()/population
        sample_size=int(1.1*num_neg_edge/prob)
        sample_size=max(sample_size,num_neg_edge) # 최소 1개 이상은 뽑도록 보정

        ### 4. Main sampling loop
        idx_cpu=idx.cpu().numpy() # CPU for np.isin
        neg_idx=None
        for _ in range(3):
            rnd=torch.randint(
                low=0,
                high=population,
                size=(sample_size,),
                device='cpu'
            )
            rnd_np=rnd.numpy()

            # 전체 edge_index에 존재하는 edge 제거
            mask=np.isin(rnd_np,idx_cpu)

            # self-loop 제거
            src_np=rnd_np//num_node
            tar_np=rnd_np%num_node
            mask|=(src_np==tar_np)

            # 이미 뽑힌 negative 제거
            if neg_idx is not None:
                mask|=np.isin(
                    rnd_np,
                    neg_idx.cpu().numpy()
                )
            mask=torch.from_numpy(mask)
            rnd=rnd[~mask]

            # batch 내부 중복 제거
            rnd=torch.unique(rnd)

            # negative 누적
            if neg_idx is None:
                neg_idx=rnd
            else:
                neg_idx=torch.cat([neg_idx,rnd],dim=0)

            # 충분히 모이면 종료
            if neg_idx.numel()>=num_neg_edge:
                neg_idx=neg_idx[:num_neg_edge]
                break

        ### 5. edge_id -> (src, tar)
        neg_src=neg_idx//num_node
        neg_tar=neg_idx%num_node
        neg_edge_index=torch.stack(
            [neg_src,neg_tar],
            dim=0
        ).to(edge_index.device)
        return neg_edge_index

    @staticmethod
    def neighbor_sampling(
            num_node:int,
            edge_index:torch.Tensor,
            pos_edge_index:torch.Tensor,
            neg_edge_index:torch.Tensor,
            n_hop:int=1
        ):
        """
        Input:
            num_node: dataset 전체 노드 수
            edge_index: [2,E], (train or val or test) edge_index
            pos_edge_indexs: batch pos_edge_index
            neg_edge_index: batch neg_edge_index
            n_hop
        Output:
            sub_edge_index
        """
        src,tar=edge_index
        pos_neg_edge_index=torch.cat(
            [pos_edge_index,neg_edge_index],
            dim=1
        )
        seed_nodes=torch.cat([
            pos_neg_edge_index[0],
            pos_neg_edge_index[1]
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

        # add self-loop for embedding
        node=torch.arange(num_node,dtype=torch.long,device=edge_index.device)
        sub_edge_index=SamplingUtils.add_self_loop_to_edge_index(
            node=node,
            edge_index=sub_edge_index
        )
        return sub_edge_index

    @staticmethod
    def get_edge_label(
            edge_index:torch.Tensor,
            label:Literal[0,1]=1,
            dtype:torch.dtype=torch.float32
        ):
        """
        Input:
            edge_index: [2,E]
            label: int
        Return:
            edge_label: [E,]
        """
        assert label in (0,1)
        edge_label=torch.full(
            size=(edge_index.size(1),),
            fill_value=label,
            dtype=dtype,
            device=edge_index.device
        )
        return edge_label