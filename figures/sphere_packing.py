import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def run():
    plt.rcParams.update({"mathtext.fontset": "cm"})

    df = pd.read_csv("data/sphere_packing.csv")

    new_df_schema = {
        'D0': df['D0'].astype(float),
        'm c0.0003': df['m c0.0003'].astype(float),
        'fc c0.0003': df['fc c0.0003'].astype(float),
        'm c0.001': df['m c0.001'].astype(float),
        'fc c0.001': df['fc c0.001'].astype(float),
        'm c0.003': df['m c0.003'].astype(float),
        'fc c0.003': df['fc c0.003'].astype(float),
        'm c0.01': df['m c0.01'].astype(float),
        'fc c0.01': df['fc c0.01'].astype(float),
        'm c0.03': df['m c0.03'].astype(float),
        'fc c0.03': df['fc c0.03'].astype(float),
        'm c0.1': df['m c0.1'].astype(float),
        'fc c0.1': df['fc c0.1'].astype(float)
    }

    df = pd.DataFrame(new_df_schema)

    concentrations = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    concentrations_str = ['0.0003','0.001', '0.003', '0.01', '0.03', '0.1']
    concentrations_str_nice = ['$3 x 10^{-4}$', '$10^{-3}$', '$3 x 10^{-3}$', '$10^{-2}$', '$3 x 10^{-2}$', '$10^{-1}$']
    num_conc = len(concentrations)

    D0s = df['D0'] # * (0.92+0.67+0.54+0.27)

    mon = pd.DataFrame([df['m c' + i] for i in concentrations_str]).transpose()
    mon.columns = concentrations_str

    fc = pd.DataFrame([df['fc c' + i] for i in concentrations_str]).transpose()
    fc.columns = concentrations_str

    cm = 'copper'
    cmap = matplotlib.cm.get_cmap(cm)
    conc_colors = [cmap(i) for i in np.linspace(0, 1, 6)]
    FS = 22

    # Plot all together
    fig = plt.figure(figsize=[7, 7])  # default [6.4, 4.8]
    ax = fig.add_subplot(111)

    for conc_i in range(num_conc):
        conc_str = concentrations_str[conc_i]
        if conc_i == 2:
            lab_m = 'Monomers'
            lab_fc = '60-mers'
        else:
            lab_m = ''
            lab_fc = ''
        ax.plot(D0s[3:], mon[conc_str][3:], '--', linewidth=3,
                color=conc_colors[conc_i], label=lab_m)
        ax.plot(D0s[3:], fc[conc_str][3:], '-', linewidth=3,
                color=conc_colors[conc_i], label=lab_fc)

    # ax.set_xlabel(r'$D_0$',fontsize=FS)
    ax.set_xlabel(r"$\epsilon/k_BT$", fontsize=FS)
    ax.set_ylabel(r'Yield',fontsize=FS)

    ax.legend(fontsize=15,frameon=True)
    ax.tick_params('both', labelsize=FS)
    plt.tight_layout()

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cm),
                        ticks=np.linspace(0,1,num_conc),
                        orientation='horizontal')
    cbar.ax.set_xticklabels(concentrations_str_nice, fontsize=15, rotation=60)
    cbar.set_label(label='Monomer Concentration', fontsize=FS)

    plt.show()
    # plt.savefig("spheres_revised.svg", format="svg")


if __name__ == "__main__":
    run()
