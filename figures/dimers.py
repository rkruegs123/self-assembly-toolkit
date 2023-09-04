import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def run():
    df = pd.read_csv("data/dimers.csv")

    new_df_schema = {
        'D0': df['D0'].astype(float),
        'yield sim 4-8 c001': df['yield sim 4-8 c001'].astype(float),
        'err sim 4-8 c001': df['err sim 4-8 c001'].astype(float),
        'yield sim 4-8 c003': df['yield sim 4-8 c003'].astype(float),
        'err sim 4-8 c003': df['err sim 4-8 c003'].astype(float)
    }

    df = pd.DataFrame(new_df_schema)

    D0s6_18 = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    Ys6_18 = np.array([2.006024410940717e-06, 1.8572365378530675e-05, 0.0001651247975901062,
                       0.002020706992934963, 0.012611001727262918, 0.20749090062530195,
                       0.6252090906474788, 0.9062405017857564, 0.9905820631679213,
                       0.9994865074570378])


    cmap = matplotlib.cm.get_cmap('copper')
    conc_colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    FS = 22
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.errorbar(df['D0'], df['yield sim 4-8 c001'], yerr=df['err sim 4-8 c001'],
                ls='none',marker='.', linewidth=1, markersize=12, label="Simulation",
                capsize=5, capthick=2, color='black')
    ax.plot(D0s6_18, Ys6_18, label='Theory', linewidth=3, color=conc_colors[5])

    ax.set_xlabel(r'$E_b$', fontsize=FS)
    ax.set_ylabel(r'Dimers Yield', fontsize=FS)
    ax.legend(loc='lower right', fontsize=15, frameon=True)
    ax.tick_params('both', labelsize=FS)

    plt.tight_layout()

    plt.show()
    plt.clf()


    def no_entropy(eb):
        return np.exp(3 * eb) / (1 + np.exp(3 * eb))

    def no_entropy_finite_conc(eb):
        return np.exp(-3 * eb) * (1000 + np.exp(3*eb) - 20*np.sqrt(2500 + 5*np.exp(3*eb)))


    all_ebs_smooth = np.linspace(0, 12, 20)
    yields1 = [no_entropy(eb) for eb in all_ebs_smooth]
    yields2 = [no_entropy_finite_conc(eb) for eb in all_ebs_smooth]

    theory_yields = [0.0, 1.46696e-07, 3.68308e-07] + list(Ys6_18)

    all_ebs = np.arange(13)


    from matplotlib import rc
    rc('text', usetex=False)


    plt.rcParams["mathtext.fontset"] = "cm"

    cmap = matplotlib.cm.get_cmap('copper')
    conc_colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    FS = 22
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=54)

    ax.errorbar(df['D0'], df['yield sim 4-8 c001'], yerr=df['err sim 4-8 c001'],
                ls='none',marker='.', linewidth=2, markersize=18,
                label="Simulation", capsize=5, capthick=2, color='black')
    ax.plot(all_ebs_smooth, yields1, label='No Entropy', linewidth=5, color=conc_colors[9])
    ax.plot(all_ebs_smooth, yields2, label='No Entropy, Finite Conc.',
            linewidth=5, color=conc_colors[5])
    ax.plot(all_ebs, theory_yields, label='Theory', linewidth=5, color=conc_colors[1])

    # ax.set_xlabel(r'$E_b/k_BT$', fontsize=28)
    ax.set_xlabel(r"$\epsilon/k_BT$", fontsize=28)
    ax.set_ylabel(r'Yield of Dimers', fontsize=28)
    ax.legend(loc='upper center', fontsize=20, bbox_to_anchor=(0.5, 1.21),
              ncol=2, fancybox=True, shadow=True)
    ax.tick_params('both', labelsize=FS)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
