from sklearn import metrics 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import seaborn as sns
import numpy as np
import copy as copy
import pdb
import os
import pandas as pd

def draw_3_separab_incremental_ood(onehots, scores, path_fig_save, cur_task_id,
                           make_plot = True,add_to_title=None,swap_classes=False):

    # auroc for old id vs ood
    onehots_old_id = onehots[onehots!=1]
    scores_old_id = scores[onehots!=1]

    auroc_old_id = metrics.roc_auc_score(onehots_old_id, scores_old_id)
    auroc_old_id = round(auroc_old_id*100, 3)

    # auroc for latest id
    onehots_latest_id = onehots[onehots!=0]
    scores_latest_id = scores[onehots!=0]

    auroc_latest_id = metrics.roc_auc_score(onehots_latest_id, scores_latest_id)
    auroc_latest_id = round(auroc_latest_id*100, 3)

    to_replot_dict = dict()

    out_scores, in_scores_latest, in_scores_old = scores[onehots==-1], scores[onehots==1], scores[onehots==0]

    plt.figure(figsize = (5.5,3),dpi=100)

    plt.title(f'Old-AUROC:{str(auroc_old_id)}, Latest-AUROC:{str(auroc_latest_id)}', fontsize=14)

    # ood samples
    vals,bins = np.histogram(out_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=4,color="navy",marker="",label="out samples")
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

    # old id samples
    vals,bins = np.histogram(in_scores_old,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=4,color="slategrey",marker="",label="old-id samples")
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="slategrey",alpha=0.3)

    # latest id samples

    vals,bins = np.histogram(in_scores_latest,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=4,color="crimson",marker="",label="latest_id samples")
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

    if make_plot:
        plt.xlabel(f"Ood score on task {cur_task_id}th",fontsize=14)
        plt.ylabel("Frequency",fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim([0,None])

        plt.legend(fontsize = 14)

        plt.tight_layout()
        plt.savefig(path_fig_save)

    return auroc_old_id, auroc_latest_id, to_replot_dict


def draw_separab_normal_ood(onehots, scores, path_fig_save,
                           make_plot = True,add_to_title=None,swap_classes=False):

        auroc = metrics.roc_auc_score(onehots, scores)

        ###
        # ind_indicator = np.zeros_like(onehots)
        # ind_indicator[onehots != -1] = 1


        # fpr, tpr, thresholds = metrics.roc_curve(ind_indicator, scores)

        # auroc1 = metrics.auc(fpr, tpr)

        ###

        to_replot_dict = dict()

        if swap_classes == False:
            out_scores,in_scores = scores[onehots==-1], scores[onehots==1] 
        else:
            out_scores,in_scores = scores[onehots==1], scores[onehots==1] 

        if make_plot:
            plt.figure(figsize = (5.5,3),dpi=100)

            if add_to_title is not None:
                plt.title(add_to_title+" AUROC="+str(float(auroc*100))[:6]+"%",fontsize=14)
            else:
                plt.title(" AUROC="+str(float(auroc*100))[:6]+"%",fontsize=14)


        vals,bins = np.histogram(out_scores,bins = 51)
        bin_centers = (bins[1:]+bins[:-1])/2.0

        if make_plot:
            plt.plot(bin_centers,vals,linewidth=4,color="navy",marker="",label="out test")
            plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

        to_replot_dict["out_bin_centers"] = bin_centers
        to_replot_dict["out_vals"] = vals

        vals,bins = np.histogram(in_scores,bins = 51)
        bin_centers = (bins[1:]+bins[:-1])/2.0

        if make_plot:
            plt.plot(bin_centers,vals,linewidth=4,color="crimson",marker="",label="in test")
            plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

        to_replot_dict["in_bin_centers"] = bin_centers
        to_replot_dict["in_vals"] = vals

        if make_plot:
            plt.xlabel(f"Ood score",fontsize=14)
            plt.ylabel("Frequency",fontsize=14)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            plt.ylim([0,None])

            plt.legend(fontsize = 14)

            plt.tight_layout()
            plt.savefig(path_fig_save)

        return auroc,to_replot_dict

def draw_joint_3_separab_incremental_ood(onehot, old_scores_list, path_save, cur_task_id):
    [prev_scores, latest_scores] = old_scores_list

    type_model = 'joint_model'

    onehot_for_auroc = copy.copy(onehot)

    onehot_for_auroc[onehot_for_auroc!=-1] = 1

    prev_auroc = metrics.roc_auc_score(onehot_for_auroc, prev_scores)
    latest_auroc = metrics.roc_auc_score(onehot_for_auroc, latest_scores)

    prev_out_scores, prev_in_old_scores, prev_in_latest_scores = prev_scores[onehot==-1], prev_scores[onehot==0], prev_scores[onehot==1]
    latest_out_scores, latest_in_old_scores, latest_in_latest_scores = latest_scores[onehot==-1], latest_scores[onehot==0], latest_scores[onehot==1]

    plt.figure(figsize = (5.5,3),dpi=100)

    
    plt.title(f"Prev AUROC={str(float(prev_auroc*100))[:6]}% vs Latest AUROC={str(float(latest_auroc*100))[:6]}%",fontsize=14)

    # for prev model
    ## ood samples
    vals,bins = np.histogram(prev_out_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="navy",marker="",label="prev model out test", linestyle='dashed')
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

    ## old id samples
    vals,bins = np.histogram(prev_in_old_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="slategrey",marker="",label="prev model in-old test", linestyle='dashed')
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="slategrey",alpha=0.3)

    ## latest id samples
    vals,bins = np.histogram(prev_in_latest_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="crimson",marker="",label="prev model in-latest test", linestyle='dashed')
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

    # for final model
    ## ood samples
    vals,bins = np.histogram(latest_out_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="navy",marker="",label="latest model out test")
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

    ## old id samples
    vals,bins = np.histogram(latest_in_old_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="slategrey",marker="",label="latest model in-old test")
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="slategrey",alpha=0.3)

    ## latest id samples
    vals,bins = np.histogram(latest_in_latest_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="crimson",marker="",label="latest model in-latest test")
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

    plt.xlabel(f"{type_model} ood score on task {cur_task_id}th",fontsize=14)
    plt.ylabel("Frequency",fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylim([0,None])

    plt.legend(fontsize = 10)

    plt.tight_layout()
    plt.savefig(path_save)


def draw_joint_separab_incremental_ood(onehots_list, old_scores_list, path_save, cur_task_id): # separability of ood, old id, and latest id samples
    [old_onehots, final_onehots] = onehots_list
    [old_scores, final_scores] = old_scores_list

    type_model = 'joint_model'

    old_auroc = metrics.roc_auc_score(old_onehots, old_scores)
    final_auroc = metrics.roc_auc_score(final_onehots, final_scores)

    to_replot_dict = dict()

    old_out_scores, old_in_old_scores, old_in_latest_scores = old_scores[old_onehots==-1], old_scores[old_onehots==0] , old_scores[old_onehots==1] 
    
    # check if final_onehots have 0 label (last task)
    final_out_scores, final_in_old_scores, final_in_latest_scores = final_scores[final_onehots==-1], final_scores[final_onehots==0], final_scores[final_onehots==1]

    pdb.set_trace()
    plt.figure(figsize = (5.5,3),dpi=100)

    
    plt.title(f"Old AUROC={str(float(old_auroc*100))[:6]}% vs Final AUROC={str(float(final_auroc*100))[:6]}%",fontsize=14)

    # for old model
    ## ood samples
    vals,bins = np.histogram(old_out_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="navy",marker="",label="old model out test", linestyle='dashed')
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

    ## old id samples
    vals,bins = np.histogram(old_in_old_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="slategray",marker="",label="old model in-old test", linestyle='dashed')
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="slategray",alpha=0.3)

    # latest id samples
    vals,bins = np.histogram(old_in_latest_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="crimson",marker="",label="old model in-latest test", linestyle='dashed')
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

    # for final model
    ## ood samples
    vals,bins = np.histogram(final_out_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="navy",marker="",label="final model out test")
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

    ## old id samples
    vals,bins = np.histogram(final_in_old_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="slategray",marker="",label="final model in-old test")
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="slategray",alpha=0.3)

    # latest id samples
    vals,bins = np.histogram(final_in_latest_scores,bins = 51)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    plt.plot(bin_centers,vals,linewidth=2,color="crimson",marker="",label="final model in-latest test", linestyle='dashed')
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

    plt.xlabel(f"{type_model} ood score on task {cur_task_id}th",fontsize=14)
    plt.ylabel("Frequency",fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylim([0,None])

    plt.legend(fontsize = 10)

    plt.tight_layout()
    plt.savefig(path_save)


def draw_tsne_each(path_save, targets, outputs):
    # old
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(outputs)

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = targets

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", len(np.unique(targets))),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(path_save, bbox_inches='tight')

    plt.clf()

def draw_tsne_incremental_ood(path_save, targets, old_outputs, final_outputs):

    tmp, _ = os.path.splitext(path_save)
    path_save_old = f'{tmp}_old.png'
    path_save_final = f'{tmp}_final.png'

    print("Start drawing tsne old")
    draw_tsne_each(path_save_old, targets, old_outputs)

    print("Start drawing tsne final")
    draw_tsne_each(path_save_final, targets, final_outputs)

