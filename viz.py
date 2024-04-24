from matplotlib import pyplot as plt
import os
import pandas as pd


def u_value_analysis(bins, u_dict, kc_set, iteration_artifacts):
    # all training observations
    u_sorted = sorted(u_dict.values())
    u_s = pd.Series(u_sorted)
    n = len(u_s)
    u_hist_100 = u_s.hist(bins=bins)
    u_hist_100 = format_hist(u_hist_100, n, u_s.min(), u_s.max())
    save_hist(u_hist_100, os.path.join(iteration_artifacts, 'u_hist_100.png'))
    # repeat, but exclude top 1% - sometimes this is better for visualization purposes)
    n_99 = int(0.99*n)
    u_s_99 = u_s[:n_99]
    u_hist_99 = u_s_99.hist(bins=bins)
    u_hist_99 = format_hist(u_hist_99, n_99, u_s_99.min(), u_s_99.max())
    save_hist(u_hist_99, os.path.join(iteration_artifacts, 'u_hist_99.png'))

    if len(kc_set) > 0:
        # training observations that are known outliers
        u_sorted = sorted([v for k, v in u_dict.items() if k in kc_set])
        u_s = pd.Series(u_sorted)
        n = len(u_s)
        u_hist_100 = u_s.hist(bins=bins)
        u_hist_100 = format_hist(u_hist_100, n, u_s.min(), u_s.max())
        save_hist(u_hist_100, os.path.join(iteration_artifacts, 'u_kc_hist_100.png'))
        # repeat, but exclude top 1% - sometimes this is better for visualization purposes)
        n_99 = int(0.99*n)
        u_s_99 = u_s[:n_99]
        u_hist_99 = u_s_99.hist(bins=bins)
        u_hist_99 = format_hist(u_hist_99, n_99, u_s_99.min(), u_s_99.max())
        save_hist(u_hist_99, os.path.join(iteration_artifacts, 'u_kc_hist_99.png'))   


def format_hist(hist, n_u, u_min, u_max):
    hist_title = '# = {n}, min u = {lo:.2E}, max u = {hi:.2E}\n'.format(n=n_u, lo=u_min, hi=u_max)
    hist.set_title(hist_title, fontsize=9)

    # add bar labels
    bars = hist.containers[0]
    hist.bar_label(bars, fontsize=5)

    return hist


def save_hist(hist, hist_path):
    fig = hist.get_figure()
    fig.tight_layout()
    fig.savefig(hist_path)
    fig.clf()
    plt.clf()
    plt.close()


def plot_learning_curves(history, iteration_artifacts, eps_tsts):
    trn_color = "red"
    val_color = "orange"

    # Accuracy
    plt.plot(history['accuracy'], color=trn_color)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.ylim((0.0, 1.0))
    legend = ['train']

    if len(history['val_acc']) > 0:
        plt.plot(history['val_acc'], color=val_color)
        legend += ['val']

    if len(eps_tsts) > 0:
        for i, eps_tst in enumerate(eps_tsts):
            tst_color = plt.cm.Blues(0.2 + i/len(eps_tsts)*0.8)
            k = 'tst_acc_' + str(eps_tst)
            plt.plot(history[k], color=tst_color)
            legend += ['tst ' + str(eps_tst)]
    else:
        plt.plot(history['tst_acc'], color="blue")
        legend += ['tst']

    plt.legend(legend, loc='lower right')
    plt.savefig(os.path.join(iteration_artifacts, 'trn_val_tst_acc.png'))
    plt.clf()
    plt.close()

    # Loss
    plt.plot(history['loss'], color=trn_color)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    legend = ['train']

    if len(history['val_loss']) > 0:
        plt.plot(history['val_loss'], color=val_color)
        legend += ['val']

    if len(eps_tsts) > 0:
        for i, eps_tst in enumerate(eps_tsts):
            tst_color = plt.cm.Blues(0.2 + i/len(eps_tsts)*0.8)
            k = 'tst_loss_' + str(eps_tst)
            plt.plot(history[k], color=tst_color)
            legend += ['tst ' + str(eps_tst)]
    else:
        plt.plot(history['tst_loss'], color="blue")
        legend += ['tst']

    plt.legend(legend, loc='upper right')
    plt.savefig(os.path.join(iteration_artifacts, 'trn_val_tst_loss.png'))
    plt.clf()
    plt.close()
    