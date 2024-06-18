import numpy as np
import pdb
import matplotlib.pyplot as plt
import os.path as osp
import seaborn as sns


# this is to analyze weight distribution of ebo on each portion approach
def draw_distribution_weight_id_task(global_wdist_sample_k, path_draw, cur_task_id):
    # first average 
    color_list = ['red', 'green', 'blue', 'grey', 'yellow', 'orange',
                  'pink', 'black', 'purple', 'olive']
    weight_list_for_each = []
    for id_task in sorted(global_wdist_sample_k.keys()):
        array = np.array(global_wdist_sample_k[id_task])
        weight_list_for_each.append(np.average(array, axis=0).tolist())
    
    # plot for each id task

    weight_axis = [f'w on {portion}th portion' for portion in range(len(weight_list_for_each))]
    for idx, each_id_task_record in enumerate(weight_list_for_each):

        fig = plt.figure(figsize=(10, 5))
        plt.bar(weight_axis, each_id_task_record, color=color_list[:len(weight_axis)])
        plt.xlabel(f"Weight distribution on {idx}th id samples")
        plt.ylabel(f"Weight")

        path_save_img = osp.join(path_draw, f'weight_dis_on_{idx}th_id_samples_at_task{cur_task_id}')
        plt.savefig(path_save_img)
        plt.close()

def draw_heatmap_mis_id_task(hm_matrix, path_draw, cur_task_id):
    path_save_img = osp.join(path_draw, f'heatmap_task{cur_task_id}')
    
    plt.imshow(hm_matrix, cmap='hot', interpolation='nearest')
    plt.savefig(path_save_img)
    plt.close()


