from torch import nn

from .registry import register_high_level_loss, get_mid_level_loss
from ..common.constant import def_logger

logger = def_logger.getChild(__name__)


class AbstractLoss(nn.Module):
    """
    An abstract loss module.

    :meth:`forward` and :meth:`__str__` should be overridden by all subclasses.

    :param sub_terms: loss module configurations.
    :type sub_terms: dict or None

    .. code-block:: YAML
       :caption: An example yaml of ``sub_terms``

        sub_terms:
          ce:
            criterion:
              key: 'CrossEntropyLoss'
              kwargs:
                reduction: 'mean'
            criterion_wrapper:
              key: 'SimpleLossWrapper'
              kwargs:
                input:
                  is_from_teacher: False
                  module_path: '.'
                  io: 'output'
                target:
                  uses_label: True
            weight: 1.0
    """
    def __init__(self, sub_terms=None, **kwargs):
        super().__init__()
        term_dict = dict()
        if sub_terms is not None:
            for loss_name, loss_config in sub_terms.items():
                sub_criterion_or_config = loss_config['criterion']
                sub_criterion = sub_criterion_or_config if isinstance(sub_criterion_or_config, nn.Module) \
                    else get_mid_level_loss(sub_criterion_or_config, loss_config.get('criterion_wrapper', None))
                term_dict[loss_name] = (sub_criterion, loss_config['weight'])
        self.term_dict = term_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function is not implemented')

    def __str__(self):
        raise NotImplementedError('forward function is not implemented')


@register_high_level_loss
class WeightedSumLoss(AbstractLoss):
    """
    A weighted sum (linear combination) of mid-/low-level loss modules.

    If ``model_term`` contains a numerical value with ``weight`` key, it will be a multiplier :math:`W_{model}`
    for the sum of model-driven loss values :math:`\sum_{i} L_{model, i}`.

    .. math:: L_{total} = W_{model} \cdot (\sum_{i} L_{model, i}) + \sum_{k} W_{sub, k} \cdot L_{sub, k}

    :param model_term: model-driven loss module configurations.
    :type model_term: dict or None
    :param sub_terms: loss module configurations.
    :type sub_terms: dict or None
    """
    def __init__(self, model_term=None, sub_terms=None, **kwargs):
        super().__init__(sub_terms=sub_terms, **kwargs)
        if model_term is None:
            model_term = dict()
        self.model_loss_factor = model_term.get('weight', None)

    def forward(self, io_dict, model_loss_dict, targets):
        loss_dict = dict()
        student_io_dict = io_dict['student']
        teacher_io_dict = io_dict['teacher']
        for loss_name, (criterion, factor) in self.term_dict.items():
            loss = criterion(student_io_dict, teacher_io_dict, targets)
            loss_dict[loss_name] = factor * loss
            #student_io_dict[loss_name] = loss.item()
            io_dict['student'][loss_name] = loss.item()

        sub_total_loss = sum(loss for loss in loss_dict.values()) if len(loss_dict) > 0 else 0
        if self.model_loss_factor is None or \
                (isinstance(self.model_loss_factor, (int, float)) and self.model_loss_factor == 0):
            return sub_total_loss

        if isinstance(self.model_loss_factor, dict):
            model_loss = sum([self.model_loss_factor[k] * v for k, v in model_loss_dict.items()])
            return sub_total_loss + model_loss
        return sub_total_loss + self.model_loss_factor * sum(model_loss_dict.values() if len(model_loss_dict) > 0 else [])

    def __str__(self):
        desc = 'Loss = '
        tuple_list = [(self.model_loss_factor, 'ModelLoss')] \
            if self.model_loss_factor is not None and self.model_loss_factor != 0 else list()
        tuple_list.extend([(factor, criterion) for criterion, factor in self.term_dict.values()])
        desc += ' + '.join(['{} * {}'.format(factor, criterion) for factor, criterion in tuple_list])
        return desc
