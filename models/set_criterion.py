import torch.nn as nn
import torch, math
from models.matcher import HungarianMatcher
import torch.nn.functional as F

class SetCriterion(nn.Module):
    """ This class computes the loss for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """
    def __init__(self, num_classes, event_type_weight = False, cost_weight = False, na_coef = 0.1, losses = ["event", "role"], matcher = 'avg'):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_weight, matcher, self.num_classes)
        self.losses = losses
        self.cost_weight = cost_weight
        if event_type_weight:
            self.type_weight = torch.tensor(event_type_weight).cuda()
        else:
            self.type_weight = torch.ones(self.num_classes).cuda()
            self.type_weight[-1] = na_coef
        self.register_buffer('rel_weight', self.type_weight)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices_tensor = self.matcher(outputs, targets)
        losses = self.get_role_loss(outputs, targets, indices_tensor)
        return losses



    def get_role_loss(self, outputs, targets, indices_tensor):

        num_sets, num_roles, num_entities = outputs["pred_role_logits"].size()

        pred_event = outputs["pred_doc_event_logps"].softmax(-1)
        gold_event = targets["doc_event_label"]
        gold_event_tensor = torch.tensor(gold_event).cuda()
        # selected_pred_event_tensor = pred_event[indices_tensor[0][0]]

        pred_role = outputs["pred_role_logits"].softmax(-1)  # [num_sets,num_roles,num_etities]
        gold_role = targets["role_label"]
        gold_role_tensor = torch.tensor(gold_role).cuda()
        # gold_role_lists = [role_list for role_list in gold_role if role_list != None]
        if self.num_classes == 2:
            gold_event_tensor = torch.zeros(gold_event_tensor.size()).long().cuda()

        selected_pred_role_tensor = pred_role[indices_tensor[0]]
        selected_gold_role_tensor = gold_role_tensor[indices_tensor[1]]
        # role_loss = self.cross_entropy(selected_pred_role_tensor.flatten(0, 1), selected_gold_role_tensor.flatten(0, 1))
        role_loss = F.cross_entropy(selected_pred_role_tensor.flatten(0, 1), selected_gold_role_tensor.flatten(0, 1))

        gold_event_label = torch.full(pred_event.shape[:1], self.num_classes -1, dtype=torch.int64).cuda()
        gold_event_label[indices_tensor[0]] = gold_event_tensor
        event_type_loss = F.cross_entropy(pred_event, gold_event_label, weight=self.type_weight)

        # print(gold_event_label, '\t', pred_event.argmax(-1))
        # print(indices_tensor)
        # print(selected_gold_role_tensor, '\n', pred_role.argmax(-1))

        losses = event_type_loss + role_loss
        # losses = role_loss
        return losses