import matplotlib.pyplot as plt


def plot_uniaxial_cauchy(F, cauchy):

    eps_xx = F[0, 0, :] - 1.
    sigma_axial = cauchy[0, 0, :]
    sigma_off_axis = cauchy[1, 1, :]

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(eps_xx, sigma_axial, color="blue", label="$\\sigma_{xx}$")
    ax.scatter(eps_xx, sigma_off_axis, color="red",
               label="$\\sigma_{yy}$ / $\\sigma_{zz}$")
    ax.set_xlabel("$\\epsilon_{xx}$", fontsize=22)
    ax.set_ylabel("Stress", fontsize=22)
    ax.set_title("$J_2$ Yield with Voce Hardening under Uniaxial Stress",
                 fontsize=22)
    ax.legend(loc="best", fontsize=18)

    return fig
