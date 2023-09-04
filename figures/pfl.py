import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def run():
    df = pd.read_csv("data/pfl.csv")

    new_df_schema = {
        'D0': df['D0'].astype(float),
        'c 0.000001': df['c 0.000001'].astype(float),
        'c 0.000003': df['c 0.000003'].astype(float),
        'c 0.00001': df['c 0.00001'].astype(float),
        'c 0.00003': df['c 0.00003'].astype(float),
        'c 0.0001': df['c 0.0001'].astype(float),
        'c 0.0003': df['c 0.0003'].astype(float)
    }

    df = pd.DataFrame(new_df_schema)

    concentrations = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003]
    concentrations_str = ['0.000001', '0.000003', '0.00001', '0.00003', '0.0001', '0.0003']
    concentrations_str_nice = ['$10^{-6}$', '$3 x 10^{-6}$', '$10^{-5}$', '$3 x 10^{-5}$', '$10^{-4}$', '$3 x 10^{-4}$']
    num_conc = len(concentrations)

    D0s = df['D0']#*(0.92+0.67+0.54+0.27)

    dim = pd.DataFrame([df['c ' + i] for i in concentrations_str]).transpose()
    dim.columns = concentrations_str

    cm = 'copper'
    cmap = matplotlib.cm.get_cmap(cm)
    conc_colors = [cmap(i) for i in np.linspace(0, 1, 6)]
    FS = 22

    # Try them all in the same plot -- I think it's too much
    fig = plt.figure(figsize=[7, 7])  # default [6.4, 4.8]
    ax = fig.add_subplot(111)

    for conc_i in range(num_conc):
        conc_str = concentrations_str[conc_i]

        if conc_i == 2:
            lab_m = 'Monomers'
            lab_d = 'Dimers'
        else:
            lab_m = ''
            lab_d = ''
        ax.plot(D0s, dim[conc_str], '-', linewidth=3, color=conc_colors[conc_i], label=lab_d)

    ax.set_xlabel(r'$D_0$', fontsize=FS)
    ax.set_ylabel(r'Dimers Yield', fontsize=FS)

    ax.tick_params('both', labelsize=FS)
    plt.tight_layout()

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cm),
                        ticks=np.linspace(0,1,num_conc),
                        orientation='horizontal')
    cbar.ax.set_xticklabels(concentrations_str_nice, fontsize=15, rotation=60)
    cbar.set_label(label='Concentration', fontsize=FS)

    plt.show()


if __name__ == "__main__":
    run()
