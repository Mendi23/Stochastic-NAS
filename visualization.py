from graphviz import Digraph
import matplotlib.pyplot as plt

def plot_genotype(geno, dir, graph_name, steps):
    g = Digraph(
        name = graph_name,
        format = 'png',
        edge_attr = dict(fontsize = '20', fontname = "times"),
        node_attr = dict(
            style = 'filled',
            shape = 'rect',
            align = 'center',
            fontsize = '20',
            height = '0.5',
            width = '0.5',
            penwidth = '2',
            fontname = "times"),
        engine = 'dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor = 'darkseagreen2')
    g.node("c_{k-1}", fillcolor = 'darkseagreen2')
    assert len(geno) % 2 == 0

    for i in range(steps):
        g.node(str(i), fillcolor = 'lightblue')

    n = 2
    start = 0
    for i in range(steps):
        end = start + n
        for k in range(start, end):
            op, j = geno[k]
            if op != 'none':
                if j == 0:
                    u = "c_{k-2}"
                elif j == 1:
                    u = "c_{k-1}"
                else:
                    u = str(j - 2)
                v = str(i)
                g.edge(u, v, label = op, fillcolor = "gray")
        n += 1
        start = end
    g.node("c_{k}", fillcolor = 'palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor = "gray")

    g.render(directory = dir, view = False, cleanup = False)


def visualize_alphas(alpha_normal, alpha_reduce, steps, multiplier, imPath, name = ""):
    ex = genotype(alpha_normal, alpha_reduce, steps, multiplier)
    plot_genotype(ex.normal, imPath, 'normal_' + name, steps)
    plot_genotype(ex.reduce, imPath, 'reduce_' + name, steps)

def visualize_training(path, acc, loss, title = "Training results"):
    fig = plt.figure()
    linewidth = 2

    ax1 = fig.add_subplot(121)
    ax1.set_title('Trainng accuracy')
    ax1.plot(np.arange(1, len(acc)+1, 1), acc, color = 'b', linewidth = linewidth)
    
    ax2 = fig.add_subplot(122)
    ax2.set_title('Trainng loss')
    ax2.plot(np.arange(1, len(loss)+1, 1), loss, color = 'r', linewidth = linewidth)
    
    for ax in [ax1, ax2]:
        ax.grid()
        ax.set(xlabel = 'epoch', ylabel = 'score')
        ax.xaxis.set_ticks(np.arange(1, len(acc)+1, 1))
    
    plt.tight_layout()
    fig.savefig(os.path.join(path, "train.png"))
