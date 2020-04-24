import torch.nn as nn
from ops import *
from torch.autograd import Variable
from genotype import PRIMITIVES

class MixedOp(nn.Module):
    """
    Mixed Operator. Formula (2) in SNAS paper
    """

    def __init__(self, C, stride):
        """
        :param C: # of filters
        :param stride: stride of conv layer
        """
        super(MixedOp, self).__init__()
        ops = (OPS[primitive](C, stride, False) for primitive in PRIMITIVES)
        self._ops = nn.ModuleList(ops)

    def forward(self, x, Z):
        return sum(z * op(x) for z, op in zip(Z, self._ops))


class Cell(nn.Module):
    """
    Cell for child network
    """

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction,
                 reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine = False)
        else:
            self.preprocess0 = ReLUConvBN(
                C_prev_prev, C, 1, 1, 0, affine = False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine = False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, Z):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for _ in range(self._steps):
            s = sum(self._ops[offset + j](h, Z[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim = 1)


class SNAS(nn.Module):
    def __init__(self,
                 C,
                 num_classes,
                 layers,
                 steps = 4,
                 multiplier = 4,
                 stem_multiplier = 3,
                 input_channels = 3):
        """
        :param C: # of filters in the first conv layer
        :param num_classes: # of predicted classes
        :param layers: # of cells to stack
        :param steps: # of intermediate nodes in a cell (in addition to 2 input and 1 output node)
        :param multiplier: output channel of a cell = multiplier * ch
        :param stem_multiplier: output channel of stem network = stem_multiplier * ch
        :param input_channles: # of input channlels
        """
        super(SNAS, self).__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.steps = steps
        self.multiplier = multiplier

        C_curr = stem_multiplier * C
        # stem network, convert input_channels to c_curr
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C_curr, 3, padding = 1, bias = False),
            nn.BatchNorm2d(C_curr))

        # c_curr is a factor of the output channels of current cell
        # output channels = multiplier * c_curr
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False

        for i in range(layers):
            # Build arch
            if i in [layers // 3, 2 * layers // 3]:
                # insert reduce cells: for layer in the middle [1/3, 2/3], reduce via stride=2
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                        reduction, reduction_prev)
            self.cells.append(cell)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input, temperature):
        s0 = s1 = self.stem(input)
        for cell in self.cells:
            alpha = self.alphas_reduce if cell.reduction else self.alphas_normal
            Z = self.architect_dist(alpha, temperature)
            s0, s1 = s1, cell(s0, s1, Z)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def architect_dist(self, alpha, temperature):
        return nn.functional.gumbel_softmax(alpha, temperature)

    def _initialize_alphas(self):

        k = sum(1 for i in range(self.steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = torch.nn.Parameter(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad = True)
        self.alphas_reduce = torch.nn.Parameter(
            1e-3 * torch.randn(k, num_ops).cuda(), requires_grad = True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def model_parameters(self):
        return (p[1] for p in self.named_parameters()
                if 'alphas_normal' not in p[0] and 'alphas_reduce' not in p[0])
