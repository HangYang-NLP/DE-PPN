"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight, matcher, num_event_type):
        super().__init__()
        self.cost_event_type = cost_weight["event_type"]
        self.cost_role = cost_weight["role"]
        self.matcher = matcher
        self.num_event_type = num_event_type

    def forward(self, outputs, targets):
        num_sets, num_roles, num_entities = outputs["pred_role_logits"].size()
        pred_event = outputs["pred_doc_event_logps"].softmax(-1)  # [num_sets, num_event_types]
        gold_event = targets["doc_event_label"]
        # gold_event_list = [gold_event_type for gold_event_type in gold_event if gold_event_type != self.num_event_type]
        gold_event_tensor = torch.tensor(gold_event).cuda()
        if self.num_event_type == 2:
            gold_event_tensor = torch.zeros(gold_event_tensor.size()).long().cuda()

        pred_role = outputs["pred_role_logits"].softmax(-1)  # [num_sets,num_roles,num_etities]
        gold_role = targets["role_label"]

        gold_role_lists = [role_list for role_list in gold_role if role_list is not None]
        # gold_roles_list = [role for role in gold_role_lists]
        gold_role = torch.tensor(gold_role_lists).cuda()

        pred_role_list = pred_role.split(1,1)
        gold_role_list = gold_role.split(1,1)

        if self.matcher == "avg":
            # cost = self.cost_role * pred_role[:, gold_role_tensor]
            cost_list = []
            for pred_role_tensor, gold_role_tensor in zip(pred_role_list, gold_role_list):
                pred_role_tensor = pred_role_tensor.squeeze(1)
                cost_list.append( -self.cost_role * pred_role_tensor[:, gold_role_tensor.squeeze(1)] )

            event_type_cost = -self.cost_event_type * pred_event[:, gold_event_tensor]
            # cost_list.append(event_type_cost)
            # cost = torch.index_select(pred_role, 1, gold_role)

        role_cost_tensor = torch.stack(cost_list)
        role_cost_tensor = role_cost_tensor.transpose(1,0)
        role_cost_tensor = role_cost_tensor.view(num_sets, num_roles, -1)
        role_cost = torch.sum(role_cost_tensor, dim=1)
        all_cost = role_cost + event_type_cost
        # all_cost = role_cost

        indices = linear_sum_assignment(all_cost.cpu().detach().numpy())
        # indices_list =  [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        indices_tensor = torch.as_tensor(indices, dtype=torch.int64)
        return indices_tensor
