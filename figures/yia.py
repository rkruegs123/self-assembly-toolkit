import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np




def runB():
    df = pd.read_csv("data/yiaB.csv")

    new_df_schema = {
        'D0': df['D0'].astype(float),
        'M 0.0003': df['M 0.0003'].astype(float),
        'N 0.0003': df['N 0.0003'].astype(float),
        'O 0.0003': df['O 0.0003'].astype(float),
        'MN 0.0003': df['MN 0.0003'].astype(float),
        'MNO 0.0003': df['MNO 0.0003'].astype(float),
        'M 0.0001': df['M 0.0001'].astype(float),
        'N 0.0001': df['N 0.0001'].astype(float),
        'O 0.0001': df['O 0.0001'].astype(float),
        'MN 0.0001': df['MN 0.0001'].astype(float),
        'MNO 0.0001': df['MNO 0.0001'].astype(float),
        'M 0.00003': df['M 0.00003'].astype(float),
        'N 0.00003': df['N 0.00003'].astype(float),
        'O 0.00003': df['O 0.00003'].astype(float),
        'MN 0.00003': df['MN 0.00003'].astype(float),
        'MNO 0.00003': df['MNO 0.00003'].astype(float),
        'M 0.00001': df['M 0.00001'].astype(float),
        'N 0.00001': df['N 0.00001'].astype(float),
        'O 0.00001': df['O 0.00001'].astype(float),
        'MN 0.00001': df['MN 0.00001'].astype(float),
        'MNO 0.00001': df['MNO 0.00001'].astype(float),
        'M 0.000003': df['M 0.000003'].astype(float),
        'N 0.000003': df['N 0.000003'].astype(float),
        'O 0.000003': df['O 0.000003'].astype(float),
        'MN 0.000003': df['MN 0.000003'].astype(float),
        'MNO 0.000003': df['MNO 0.000003'].astype(float),
        'M 0.000001': df['M 0.000001'].astype(float),
        'N 0.000001': df['N 0.000001'].astype(float),
        'O 0.000001': df['O 0.000001'].astype(float),
        'MN 0.000001': df['MN 0.000001'].astype(float),
        'MNO 0.000001': df['MNO 0.000001'].astype(float)
    }

    df = pd.DataFrame(new_df_schema)

    concentrations = [0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003]
    concentrations_str = ['0.000001', '0.000003','0.00001', '0.00003', '0.0001', '0.0003']
    concentrations_str_nice = ['$10^{-6}$', '$3 x 10^{-6}$', '$10^{-5}$', '$3 x 10^{-5}$', '$10^{-4}$', '$3 x 10^{-4}$']
    num_conc = len(concentrations)

    D0s = df['D0']

    M = pd.DataFrame([df['M ' + i] for i in concentrations_str]).transpose()
    M.columns = concentrations_str

    N = pd.DataFrame([df['N ' + i] for i in concentrations_str]).transpose()
    N.columns = concentrations_str

    O = pd.DataFrame([df['O ' + i] for i in concentrations_str]).transpose()
    O.columns = concentrations_str

    MN = pd.DataFrame([df['MN ' + i] for i in concentrations_str]).transpose()
    MN.columns = concentrations_str

    MNO = pd.DataFrame([df['MNO ' + i] for i in concentrations_str]).transpose()
    MNO.columns = concentrations_str


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
            lab_M = 'M'
            lab_N = 'N'
            lab_O = 'O'
            lab_MpNpO = 'M+N+O'
            lab_MNpO = 'MN+O'
            lab_MN = 'MN'
            lab_MNO = 'MNO'
        else:
            lab_M = ''
            lab_N = ''
            lab_O = ''
            lab_MpNpO = ''
            lab_MNpO = ''
            lab_MN = ''
            lab_MNO = ''

        ax.plot(D0s, M[conc_str]/(M[conc_str]+MN[conc_str]+MNO[conc_str]), ':',
                linewidth=3, color=conc_colors[conc_i], label=lab_MpNpO)
        ax.plot(D0s, MN[conc_str]/(M[conc_str]+MN[conc_str]+MNO[conc_str]), '--',
                linewidth=3, color=conc_colors[conc_i], label=lab_MNpO)
        ax.plot(D0s, MNO[conc_str]/(M[conc_str]+MN[conc_str]+MNO[conc_str]), '-',
                linewidth=3, color=conc_colors[conc_i], label=lab_MNO)


    ax.set_xlabel(r'$D_0$', fontsize=FS)
    ax.set_ylabel(r'$c_s/c_M^{tot}$', fontsize=FS)
    ax.set_xlim([1.5, 6.5])

    ax.legend(fontsize=15, frameon=True)
    ax.tick_params('both', labelsize=FS)
    plt.tight_layout()


    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cm),
                        ticks=np.linspace(0, 1, num_conc),
                        orientation='horizontal')
    cbar.ax.set_xticklabels(concentrations_str_nice, fontsize=15, rotation=60)
    cbar.set_label(label='$c_M^{tot}$', fontsize=FS)

    plt.show()


def runC():
    df = pd.read_csv("data/yiaC.csv")

    new_df_schema = {
        'DX': df['DX'].astype(float),
        'M': df['M'].astype(float),
        'N': df['N'].astype(float),
        'O': df['O'].astype(float),
        'MN': df['MN'].astype(float),
        'NO': df['NO'].astype(float),
        'MO': df['MO'].astype(float),
        'MNO': df['MNO'].astype(float),
        'DZ': df['DZ'].astype(float),
        'DZ*DMN': df['DZ*DMN'].astype(float),
        'DX*DMO': df['DX*DMO'].astype(float),
        'DX*DMO/DZ*DMN': df['DX*DMO/DZ*DMN'].astype(float),
        'DX/DZ': df['DX/DZ'].astype(float)
    }

    df = pd.DataFrame(new_df_schema)


    cm = 'copper'
    cmap = matplotlib.cm.get_cmap(cm)
    conc_colors = [cmap(i) for i in np.linspace(0, 1, 7)]
    FS = 22

    lab_M = 'M'
    lab_N = 'N'
    lab_O = 'O'
    lab_MpNpO = 'M+N+O'
    lab_MNpO = 'MN+O'
    lab_MOpN = 'MO+N'
    lab_MN = 'MN'
    lab_MO = 'MO'
    lab_NO = 'NO'
    lab_MNO = 'MNO'

    # Vary DX
    fig = plt.figure(figsize=[7, 5])  # default [6.4, 4.8]
    ax = fig.add_subplot(111)

    ax.plot(df['DX'], df['M']/(df['M']+df['MO']+df['MN']+df['MNO']),
            linewidth=4, color=conc_colors[0], label=lab_MpNpO)
    ax.plot(df['DX'], df['MO']/(df['M']+df['MO']+df['MN']+df['MNO']),
            linewidth=4, color=conc_colors[2], label=lab_MOpN)
    ax.plot(df['DX'], df['MN']/(df['M']+df['MO']+df['MN']+df['MNO']),
            linewidth=4, color=conc_colors[4], label=lab_MNpO)
    ax.plot(df['DX'], df['MNO']/(df['M']+df['MO']+df['MN']+df['MNO']),
            linewidth=4, color=conc_colors[6], label=lab_MNO)


    ax.set_xlabel(r'$D_X$', fontsize=FS)
    ax.set_ylabel(r'$c_s/c_M^{tot}$', fontsize=FS)

    ax.legend(fontsize=15, frameon=True)
    ax.tick_params('both', labelsize=FS)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    runB()
    runC()
