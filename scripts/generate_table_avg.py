import sys
import os
import pickle
import numpy as np
from scipy import stats
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plots import (
    winrate,
    benchs,
    splits,
    methods,
    # number_items,
    agg_metric,
    load_scores,
    make_table_avg
)
from utils import (
    dump_pickle,
)
sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename_suffix_mmlu_fields', type=str, help='path suffix', default='')
    parser.add_argument('--scenarios_to_skip', type=str, help='scenarios to skip', default='')
    parser.add_argument('--ordered', action='store_true', help='ordered', default=False)
    parser.add_argument('--agg_type', type=str, help='agg type', default='acc', choices=['acc', 'rank'])
    args = parser.parse_args()
    filename_suffix_mmlu_fields = args.filename_suffix_mmlu_fields

    table_avg = {}
    table_std = {}
    model_perf = {}

    for bench in benchs:

        table_avg[bench] = {}
        table_std[bench] = {}
        model_perf[bench] = {}
        for split in splits[bench]:
            table_avg[bench][split] = {}
            table_std[bench][split] = {}
            model_perf[bench][split] = {}
            if bench == 'mmlu_fields' and split == 'iid':
                filename_suffix = filename_suffix_mmlu_fields
            else:
                filename_suffix = ''

            full_results_path = f'results/accs_{bench}_split-{split}_iterations-5{filename_suffix}.pickle'
            with open(full_results_path, 'rb') as handle:
                data = pickle.load(handle)
            ordered = args.ordered if bench == "mmlu_fields" else False
            current_table_avg, current_table_std, current_model_perf = make_table_avg(
                bench,
                split,
                filename_suffix,
                data,
                scenarios_to_skip=args.scenarios_to_skip.split(','),
                ordered=ordered,
                return_perf_table=True,
                agg_type=args.agg_type
            )
            # print("DEBUG: current_table_avg", current_table_avg[bench][split].keys())
            table_avg[bench][split] = current_table_avg[bench][split]
            table_std[bench][split] = current_table_std[bench][split]
            model_perf[bench][split] = current_model_perf[bench][split]

    # ??
    # agg = 'leaderboard' # 'leaderboard', 'scenarios'
    # results = 'acc'# 'acc', 'rank'

    # if results == 'acc': ylim = (0,.1)
    # elif results == 'rank':
    #     if agg_metric == 'std': ylim = (0,.1)
    #     else: ylim = (.5,1)
    # else: raise NotImplementedError

    # table_avg = {}
    # table_std = {}
    # model_perf = {}
    # for bench in benchs:
    #     table_avg[bench] = {}
    #     table_std[bench] = {}
    #     model_perf[bench] = {}

    #     for split in splits[bench]:
    #         table_avg[bench][split] = {}
    #         table_std[bench][split] = {}
    #         model_perf[bench][split] = {}

    #         if bench == 'mmlu_fields' and split == 'iid':
    #             filename_suffix = filename_suffix_mmlu_fields
    #         else:
    #             filename_suffix = ''

    #         full_results_path = f'results/accs_{bench}_split-{split}_iterations-5{filename_suffix}.pickle'

    #         with open(full_results_path, 'rb') as handle:
    #             data = pickle.load(handle)

    #         models = list(data.keys())
    #         number_items = list(data[models[0]].keys())
    #         methods = list(data[models[0]][number_items[0]].keys())
    #         scenarios = list(data[models[0]][number_items[0]][methods[0]].keys())

    #         # DEBUG: when can not create np array
    #         # for method in methods:
    #         #     for number_item in number_items:
    #         #         for model in models:
    #         #             for scenario in scenarios:
    #         #                 print(model, number_item, method, scenario, len(data[model][number_item][method][scenario]))

    #         data = np.array([[[[data[model][number_item][method][scenario] for scenario in scenarios]  for model in data.keys()] for number_item in number_items] for method in methods])
    #         scores = load_scores(bench, split, scenarios_to_skip=args.scenarios_to_skip.split(','))

    #         if agg == 'leaderboard':
    #             if bench=='helm':
    #                 ###
    #                 if results == 'acc':
    #                     ###
    #                     model_perf[bench][split]['truth'] = winrate(scores, axis=1).mean(axis=0)
    #                     for i,method in enumerate(methods):
    #                         model_perf[bench][split][method] = {}
    #                         model_perf[bench][split][method] = {}
    #                     for j,number_item in enumerate(number_items):
    #                         model_perf[bench][split][method][number_item] = winrate(data, axis=2).mean(axis=3)[i,j,:,:]
    #                     ###
    #                     data = np.abs(winrate(data, axis=2).mean(axis=3)-winrate(scores, axis=1).mean(axis=0)[None,None,:,None])
    #                 elif results == 'rank':
    #                     rank_corrs = np.zeros(data.mean(axis=2).mean(axis=2).shape)
    #                     #print(bench,rank_corrs.shape)
    #                     for i in range(rank_corrs.shape[0]):
    #                         for j in range(rank_corrs.shape[1]):
    #                             for l in range(rank_corrs.shape[2]):
    #                                 #print(winrate(data, axis=2).mean(axis=3).shape)
    #                                 rank_corrs[i,j,l] = stats.spearmanr(winrate(data, axis=2).mean(axis=3)[i,j,:,l], winrate(scores.T, axis=0).mean(axis=1)).statistic
    #                     data=rank_corrs

    #                 else:
    #                     raise NotImplementedError
    #             else:
    #                 ###
    #                 if results == 'acc':
    #                     # ###
    #                     model_perf[bench][split]['truth'] = scores.mean(axis=0)
    #                     for i,method in enumerate(methods):
    #                         model_perf[bench][split][method] = {}
    #                         model_perf[bench][split][method] = {}
    #                         for j,number_item in enumerate(number_items):
    #                             model_perf[bench][split][method][number_item] = data.mean(axis=3)[i,j,:,:]
    #                     # ###
    #                     data = np.abs(data.mean(axis=3)-scores.mean(axis=0)[None,None,:,None])
    #                 elif results == 'rank':
    #                     rank_corrs = np.zeros(data.mean(axis=2).mean(axis=2).shape)
    #                     #print(bench,rank_corrs.shape)
    #                     for i in range(rank_corrs.shape[0]):
    #                         for j in range(rank_corrs.shape[1]):
    #                             for l in range(rank_corrs.shape[2]):
    #                                 #print(data.mean(axis=3).shape)
    #                                 rank_corrs[i,j,l] = stats.spearmanr(data.mean(axis=3)[i,j,:,l], scores.T.mean(axis=1)).statistic
    #                     data=rank_corrs
    #                 else:
    #                     raise NotImplementedError
    #         elif agg == 'scenarios':
    #             if results == 'acc':
    #                 data = np.abs(data-scores.T[None,None,:,:,None]).mean(axis=3)
    #             elif results == 'rank':
    #                 rank_corrs = np.zeros(data.mean(axis=2).shape)
    #                 for i in range(rank_corrs.shape[0]):
    #                     for j in range(rank_corrs.shape[1]):
    #                         for k in range(rank_corrs.shape[2]):
    #                             for l in range(rank_corrs.shape[3]):
    #                                 rank_corrs[i,j,k,l] = stats.spearmanr(data[i,j,:,k,l], scores.T[:,k]).statistic
    #                 data=rank_corrs
    #             else:
    #                 raise NotImplementedError
    #         else:
    #             raise NotImplementedError

    #         if agg_metric=='avg':
    #             data = data.mean(-1) #iterations
    #         elif agg_metric=='std':
    #             data = data.std(-1)
    #         else:
    #             raise NotImplementedError


    #         for i,method in enumerate(methods):
    #             table_avg[bench][split][method] = {}
    #             table_std[bench][split][method] = {}

    #             for j,number_item in enumerate(number_items):
    #                 if agg == 'leaderboard' and results == 'rank':
    #                     #print(data.shape)
    #                     table_avg[bench][split][method][number_item] = data[i,j]
    #                     table_std[bench][split][method][number_item] = 0
    #                 else:
    #                     #print(data.shape)
    #                     table_avg[bench][split][method][number_item] = np.mean(data, axis=-1)[i,j]
    #                     table_std[bench][split][method][number_item] = data.std(-1)[i,j]

        # with open('results/table_avg.pickle', 'wb') as handle:
        #     pickle.dump(table_avg, handle)
    dump_pickle(table_avg, 'results/table_avg.pickle')

    dump_pickle(table_std, 'results/table_std.pickle')

    dump_pickle(model_perf, 'results/model_perf.pickle')


if __name__ == '__main__':
    main()
