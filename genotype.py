from collections import namedtuple

PRIMITIVES = [
  'none',
  'max_pool_3x3',
  'avg_pool_3x3',
  'skip_connect',
  'sep_conv_3x3',
  'sep_conv_5x5',
  'dil_conv_3x3',
  'dil_conv_5x5'
]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def genotype(alpha_normal, alpha_reduce, steps, multiplier):
    def _parse(weights):
        ops = (PRIMITIVES[x] for x in weights.max(dim=1)[1].tolist())
        edges = (n for i in range(steps) for n in range(2 + i))
        return list(zip(ops, edges))
        
    gene_normal = _parse(alpha_normal)
    gene_reduce = _parse(alpha_reduce)

    concat = range(2+steps-multiplier, steps+2)
    geno = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return geno
