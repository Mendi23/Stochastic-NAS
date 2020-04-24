# SNAS
[SNAS: STOCHASTICS NEURAL ARCHITECTURE SEARCH](https://arxiv.org/abs/1812.09926)

Pytorch implementation of the proposed SNAS pipeline, automated framework for neural architucture search.
full details can be found in the paper. this is **not** the official implementation


**Paremeters**

Default value for all parameters is as described in the paper. I used DARTS's default values for those which weren't mentioned.

all parameteres can be modified in `main.py`.

When resuming training from a saved checkpoint, don't change the parameters in `main.py` unless you **really** knows what you're doing.

**Reference code:**
Since the SNAS pipeline closely resembles the one of DARTS, I based this on the orginal DARTS implementation:
- https://github.com/quark0/darts

While coding, I found two other implementations on which I leaned heavily, espeecially for validity comparison, enabling parralel training, etc.:
- https://github.com/Astrodyn94/SNAS-Stochastic-Neural-Architecture-Search-
- https://github.com/JunrQ/NAS/blob/master/snas/snas


**Runing example**

Can use `enviourment.yml` to create the required conda environment or install the specified packages manually. 

```shell
python main.py
```

**Features:**
- Support multi-gpus training
- Support preempting and restarting training sessions
- Visualising both training stats and the discovered cells

*doesn't work. yet*
- reproduce the results when using a fixed seed

*to add*
- Resources constraints, as described in the paper
