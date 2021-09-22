import logging
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import binary_metric as bm
import configuration as config
# import scipy.stats as stats
import utils

logging.basicConfig(
    level=logging.INFO # allow DEBUG level messages to pass through the logger
    )

def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.
    Ex:
    ['1','10','2'] -> ['1','2','10']
    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']
    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


def print_latex_tables(df, eval_dir, circle=False):
    """
    Report geometric measures in latex tables to be used in the ACDC challenge paper.
    Prints mean (+- std) values for Dice for all structures.
    :param df:
    :param eval_dir:
    :return:
    """
    if not circle:
        out_file = os.path.join(eval_dir, 'latex_tables.txt')
    else:
        out_file = os.path.join(eval_dir, 'circle_latex_tables.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('table 1\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # prints mean (+- std) values for Dice, all structures, averaged over both phases.

        header_string = ' & '
        line_string = 'METHOD '


        for s_idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
            for measure in ['dice']:

                header_string += ' & {} ({}) '.format(measure, struc_name)

                dat = df.loc[df['struc'] == struc_name]

                if measure == 'dice':
                    line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                else:
                    line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

            if s_idx < 2:
                header_string += ' & '
                line_string += ' & '

        header_string += ' \\\\ \n'
        line_string += ' \\\\ \n'

        text_file.write(header_string)
        text_file.write(line_string)


        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('table 2\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # table 2: mean (+- std) values for Dice and HD, all structures, both phases separately


        for idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
            # new line
            header_string = ' & '
            line_string = '({}) '.format(struc_name)

            for p_idx, phase in enumerate(['ED', 'ES']):
                for measure in ['dice', 'hd']:

                    header_string += ' & {} ({}) '.format(phase, measure)

                    dat = df.loc[(df['phase'] == phase) & (df['struc'] == struc_name)]

                    if measure == 'dice':
                        line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                    else:
                        line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

                if p_idx == 0:
                    header_string += ' & '
                    line_string += ' & '

            header_string += ' \\\\ \n'
            line_string += ' \\\\ \n'

            if idx == 0:
                text_file.write(header_string)

            text_file.write(line_string)

    return 0


def compute_metrics_on_directories_raw(input_fold, output_fold, dice=True):
    '''
    - Dice
    - Hausdorff distance
    - Sens, Spec, F1
    - Predicted volume
    - Volume error w.r.t. ground truth
    :return: Pandas dataframe with all measures in a row for each prediction and each structure
    '''
    if dice:
        data = h5py.File(os.path.join(input_fold, 'pred_on_dice.hdf5'), 'r')
    else:
        data = h5py.File(os.path.join(input_fold, 'pred_on_loss.hdf5'), 'r')
        
    cardiac_phase = []
    file_names = []
    structure_names = []

    # measures per structure:
    dices_list = []
    hausdorff_list = []
    vol_list = []
    vol_err_list = []
    vol_gt_list = []
    
    dices_cir_list = []
    hausdorff_cir_list = []
    vol_cir_list = []
    vol_err_cir_list = []
    
    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
    
    for paz in np.unique(data['paz'][:]):
        ind = np.where(data['paz'][:] == paz)
        
        for ph in np.unique(data['phase'][:]):
            
            pred_arr = []  #predizione del modello
            mask_arr = []  #ground trouth
            cir_arr = []  #predizione circle
            
            for i in range(len(ind[0])):
                if data['phase'][ind[0][i]] == ph:
                    pred_arr.append(data['pred'][ind[0][i]])
                    mask_arr.append(data['mask'][ind[0][i]])
                    cir_arr.append(data['mask_cir'][ind[0][i]])

            pred_arr = np.transpose(np.asarray(pred_arr, dtype=np.uint8), (1,2,0))
            mask_arr = np.transpose(np.asarray(mask_arr, dtype=np.uint8), (1,2,0))
            cir_arr = np.transpose(np.asarray(cir_arr, dtype=np.uint8), (1,2,0))
            
            for struc in [3,1,2]:
                gt_binary = (mask_arr == struc) * 1
                pred_binary = (pred_arr == struc) * 1
                cir_binary = (cir_arr == struc) * 1
                
                #vol[ml] = n_pixel * (x_dim*y_dim) * z_dim / 1000
                # 1 mm^3 = 0.001 ml
                volpred = pred_binary.sum() * (1*1) * config.z_dim / 1000.
                volgt = gt_binary.sum() * (1*1) * config.z_dim / 1000.
                volcir = cir_binary.sum() * (1*1) * config.z_dim / 1000
            
                vol_list.append(volpred)  #volume predetto CNN
                vol_cir_list.append(volcir)  # volume predetto circle
                vol_err_list.append(volpred - volgt)  
                vol_err_cir_list.append(volcir -volgt)
                vol_gt_list.append(volgt)  #volume reale
                
                if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                    dices_list.append(1)
                    hausdorff_list.append(0)
                elif np.sum(pred_binary) == 0 and np.sum(gt_binary) > 0:
                    #logging.warning('Structure missing in either GT (x)or prediction. HD will not be accurate.')
                    dices_list.append(0)
                    hausdorff_list.append(1)
                elif np.sum(pred_binary) != 0 and np.sum(gt_binary) != 0:
                    hausdorff_list.append(bm.hd(gt_binary, pred_binary, connectivity=1))
                    dices_list.append(bm.dc(gt_binary, pred_binary))
                #np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0
                 
                if np.sum(gt_binary) == 0 and np.sum(cir_binary) == 0:
                    dices_cir_list.append(1)
                    hausdorff_cir_list.append(0)
                elif np.sum(cir_binary) == 0 and np.sum(gt_binary) > 0:
                    #logging.warning('Structure missing in either GT (x)or prediction. HD will not be accurate.')
                    dices_cir_list.append(0)
                    hausdorff_cir_list.append(1)
                elif np.sum(cir_binary) != 0 and np.sum(gt_binary) != 0:
                    hausdorff_cir_list.append(bm.hd(gt_binary, cir_binary, connectivity=1))
                    dices_cir_list.append(bm.dc(gt_binary, cir_binary))

                cardiac_phase.append(ph)
                file_names.append(paz)
                structure_names.append(structures_dict[struc])
    
    #CNN
    df1 = pd.DataFrame({'dice': dices_list, 'hd': hausdorff_list,
                       'vol': vol_list, 'vol_gt': vol_gt_list, 'vol_err': vol_err_list,
                       'phase': cardiac_phase, 'struc': structure_names, 'filename': file_names})
    
    #Circle
    df2 = pd.DataFrame({'dice': dices_cir_list, 'hd': hausdorff_cir_list,
                       'vol': vol_cir_list, 'vol_gt': vol_gt_list, 'vol_err': vol_err_cir_list,
                       'phase': cardiac_phase, 'struc': structure_names, 'filename': file_names})
    
    data.close()
    return df1, df2
    

def print_stats(df, eval_dir, circle=False):
    
    if not circle:
        out_file = os.path.join(eval_dir, 'summary_report.txt')
    else:
        out_file = os.path.join(eval_dir, 'circle_summary_report.txt')
    
    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Summary of geometric evaluation measures. \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for struc_name in ['LV', 'RV', 'Myo']:

            text_file.write(struc_name)
            text_file.write('\n')

            for cardiac_phase in ['ED', 'ES']:

                text_file.write('    {}\n'.format(cardiac_phase))

                dat = df.loc[(df['phase'] == cardiac_phase) & (df['struc'] == struc_name)]

                for measure_name in ['dice', 'hd']:

                    text_file.write('       {} -- mean (std): {:.3f} ({:.3f}) \n'.format(measure_name,
                                                                         np.mean(dat[measure_name]), np.std(dat[measure_name])))

                    ind_med = np.argsort(dat[measure_name]).iloc[len(dat[measure_name])//2]
                    text_file.write('             median {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_med], dat['filename'].iloc[ind_med]))

                    ind_worst = np.argsort(dat[measure_name]).iloc[0]
                    text_file.write('             worst {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_worst], dat['filename'].iloc[ind_worst]))

                    ind_best = np.argsort(dat[measure_name]).iloc[-1]
                    text_file.write('             best {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_best], dat['filename'].iloc[ind_best]))


        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Correlation between prediction and ground truth\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for struc_name in ['LV', 'RV']:

            lv = df.loc[df['struc'] == struc_name]

            ED_vol = np.array(lv.loc[lv['phase'] == 'ED']['vol'])
            ES_vol = np.array(lv.loc[(lv['phase'] == 'ES')]['vol'])
            SV_pred = ED_vol - ES_vol
            EF_pred = SV_pred / ED_vol

            ED_vol_gt = ED_vol - np.array(lv.loc[lv['phase'] == 'ED']['vol_gt'])
            ES_vol_gt = ES_vol - np.array(lv.loc[(lv['phase'] == 'ES')]['vol_gt'])
            SV_gt = ED_vol_gt - ES_vol_gt
            EF_gt = SV_gt / ED_vol_gt

            #EF_corr, _ = stats.pearsonr(EF_pred, EF_gt)
            EF_corr = (np.cov(EF_pred,EF_gt)[1,0] / (np.std(EF_pred) * np.std(EF_gt)) )
            SV_corr = (np.cov(SV_pred,SV_gt)[1,0] / (np.std(SV_pred) * np.std(SV_gt)) )
                       
            text_file.write('{}, SV pred: {}, SV gt: {}, corr: {}\n\n'.format(struc_name, SV_pred, SV_gt, SV_corr))
            text_file.write('{}, EF pred: {}%, EF gt: {}%, corr: {}\n\n'.format(struc_name, EF_pred*100, EF_gt*100, EF_corr))
            


def boxplot_metrics(df, eval_dir, circle=False):
    """
    Create summary boxplots of all geometric measures.
    :param df:
    :param eval_dir:
    :param circle: 
    :return:
    """
    if not circle:
        boxplots_file = os.path.join(eval_dir, 'boxplots.eps')
    else:
        boxplots_file = os.path.join(eval_dir, 'circle_boxplots.eps')

    fig, axes = plt.subplots(2, 1)
    fig.set_figheight(14)
    fig.set_figwidth(7)

    sns.boxplot(x='struc', y='dice', hue='phase', data=df, palette="PRGn", ax=axes[0])
    sns.boxplot(x='struc', y='hd', hue='phase', data=df, palette="PRGn", ax=axes[1])
    
    plt.savefig(boxplots_file)
    plt.close()

    return 0
    

def main(path_pred):
    logging.info(path_pred)
    
    if os.path.exists(os.path.join(path_pred, 'pred_on_dice.hdf5')):
        path_eval = os.path.join(path_pred, 'eval_on_dice')
        if not os.path.exists(path_eval):
            utils.makefolder(path_eval)
            logging.info(path_eval)
            df1, df2 = compute_metrics_on_directories_raw(path_pred, path_eval, dice=True)
            
            print_stats(df1, path_eval, circle=False)
            print_latex_tables(df1, path_eval, circle=False)
            boxplot_metrics(df1, path_eval, circle=False)
            print_stats(df2, path_eval, circle=True)
            print_latex_tables(df2, path_eval, circle=True)
            boxplot_metrics(df2, path_eval, circle=True)
            
            logging.info('------------Average Dice Figures----------')
            logging.info('Dice 1: %f' % np.mean(df1.loc[df1['struc'] == 'LV']['dice']))
            logging.info('Dice 2: %f' % np.mean(df1.loc[df1['struc'] == 'RV']['dice']))
            logging.info('Dice 3: %f' % np.mean(df1.loc[df1['struc'] == 'Myo']['dice']))
            logging.info('Mean dice: %f' % np.mean(np.mean(df1['dice'])))
            logging.info('------------------------------------------')
    
    if os.path.exists(os.path.join(path_pred, 'pred_on_loss.hdf5')):
        path_eval = os.path.join(path_pred, 'eval_on_loss')
        if not os.path.exists(path_eval):
            utils.makefolder(path_eval)
            logging.info(path_eval)
            df1, df2 = compute_metrics_on_directories_raw(path_pred, path_eval, dice=False)
            
            print_stats(df1, path_eval, circle=False)
            print_latex_tables(df1, path_eval, circle=False)
            boxplot_metrics(df1, path_eval, circle=False)
            print_stats(df2, path_eval, circle=True)
            print_latex_tables(df2, path_eval, circle=True)
            boxplot_metrics(df2, path_eval, circle=True)
            
            logging.info('------------Average Dice Figures----------')
            logging.info('Dice 1: %f' % np.mean(df1.loc[df1['struc'] == 'LV']['dice']))
            logging.info('Dice 2: %f' % np.mean(df1.loc[df1['struc'] == 'RV']['dice']))
            logging.info('Dice 3: %f' % np.mean(df1.loc[df1['struc'] == 'Myo']['dice']))
            logging.info('Mean dice: %f' % np.mean(np.mean(df1['dice'])))
            logging.info('------------------------------------------')
