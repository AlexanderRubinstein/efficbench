import pickle
import copy
import pandas as pd
import argparse
from scipy import stats
import os
import numpy as np
# from experiments import *
from experiments import (
    evaluate_scenarios
)
from utils import (
    lb_scenarios,
    # get_lambda,
    # SuppressPrints,
    # sigmoid,
    # item_curve,
    # item_response_function,
    # prepare_data,
    dump_pickle,
    load_pickle,
    alpaca_scenarios,
    icl_templates_scenarios
)
from plots import (
    winrate,
    benchs,
    splits,
    methods,
    # number_items,
    agg_metric,
    load_scores,
    make_perf_table
)
from generating_data.utils_for_notebooks import merge_methods
from stnd.utility.utils import apply_random_seed


SCENARIOS_TO_SKIP = ['harness_gsm8k_5']
MAX_TABLE_SIZE = 1000
RANDOM_SEED = 42

def get_data(bench, split):
    # Loading data
    if bench in ['lb','mmlu']:
        #data
        with open('data/lb.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = {'mmlu': lb_scenarios['mmlu']} if bench == 'mmlu' else lb_scenarios

        #split
        if split == 'iid':
            set_of_rows = [list(range(0,len(data['models']),4))]
        elif split == 'noniid':
            set_of_rows = [list(range(int(len(data['models'])/4))),]
        elif split == 'noniid2':
            set_of_rows = [list(range(200))]
        elif split == 'noniid3':
            set_of_rows = [list(range(300))]

        print(len(set_of_rows[0]), len(data['models']))

    elif bench == 'helm_lite':
        #data
        with open('data/helm_lite.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = helm_lite_scenarios

        #split
        if split == 'iid':
            set_of_rows = [[0,11,22],
                           [1,12,23],
                           [2,13,24],
                           [3,14,25],
                           [4,15,26],
                           [5,16,27],
                           [6,17,28],
                           [7,18,29],
                           [8,19],
                           [9,20],
                           [10,21]]
        else:
            set_of_rows = [[0,1], #AI: Yi
                           [2,3,4], #AlephAlpha_luminous
                           [5,6], #ai21_j2
                           [7,8,9,10], #anthropic_claude
                           [11,12],#cohere
                           [13,14], #google
                           [15,16,17,18], #llama
                           [19,20], #mistral ai
                           [21,22,23,24,25], #openai
                           [26,27], #TII/UAE
                           [28,29]] #writer

        print(len(set_of_rows[0]), len(data['models']))

    elif bench == 'alpaca':
        #data
        with open('data/alpaca_v2.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = alpaca_scenarios

        #split
        if split == 'iid':
            set_of_rows = [list(range(0,len(data['models']),4)),
                           list(range(1,len(data['models'])+1,4)),
                           list(range(2,len(data['models'])+2,4)),
                           list(range(3,len(data['models'])+3,4))]
        elif split == 'noniid':
            set_of_rows = [list(range(int(len(data['models'])/4))),]
        elif split == 'noniid2':
            set_of_rows = [list(range(50))]
        elif split == 'noniid3':
            set_of_rows = [list(range(75))]

        print(len(set_of_rows[0]), len(data['models']))

    # Loading data
    elif bench == 'mmlu_fields':

        #data
        with open('data/mmlu_fields.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = lb_scenarios
        scenarios = {'mmlu':scenarios['mmlu']}

        #split
        if split == 'iid':
            k = int(len(data['models'])/40)
            set_of_rows = [list(range(0,len(data['models']),k))]
        else:
            set_of_rows = [list(range(40))]
        print(len(set_of_rows[0]), len(data['models']))

    elif bench == 'icl_templates':
        #data
        with open('data/icl_templates.pickle', 'rb') as handle:
            data = pickle.load(handle)

        #scenarios
        scenarios = icl_templates_scenarios

        #split
        if split == 'iid':
            import random
            random.seed(42) #0
            list1 = random.sample(range(len(data['models'])), int(len(data['models'])/2))
            list2 = [i for i in range(len(data['models'])) if i not in list1]
            set_of_rows = [list1, list2]

        elif split == 'noniid': #instruction
            templates = [['GPT_3_style','MNLI_crowdsource','always_sometimes_never',
                          'based_on_the_previous_passage','can_we_infer','claim_true_false_inconclusive',
                          'consider_always_sometimes_never','does_it_follow_that'],['does_this_imply',
                          'guaranteed_possible_impossible', 'guaranteed_true', 'justified_in_saying',
                          'must_be_true','should_assume','take_the_following_as_truth']]
            set_of_rows = [[i for i,m in enumerate(data['models']) if np.sum([t in m for t in temp])>0] for temp in templates]

        elif split == 'noniid2': #size
            sizes = [['65b']]
            set_of_rows = [[i for i,m in enumerate(data['models']) if np.sum([t in m for t in size])>0] for size in sizes]

        elif split == 'noniid3': #same vs cross instr
            cross = [['same_instr'],['cross_instr']]
            set_of_rows = [[i for i,m in enumerate(data['models']) if np.sum([t in m for t in cr])>0] for cr in cross]

        else:
            raise NotImplementedError

        print(len(set_of_rows[0]), len(data['models']))

    else:
        raise NotImplementedError

    return data, scenarios, set_of_rows


def main():

    # User input
    parser = argparse.ArgumentParser(description='Example script with named arguments.')

    parser.add_argument('--bench', type=str, help='Benchmark (helm_lite, lb, mmlu, alpaca, icl_templates)', default='lb')
    parser.add_argument('--split', type=str, help='iid/noniid/noniid2/noniid3', default='iid')
    parser.add_argument('--iterations', type=int, help='iterations', default=3)
    parser.add_argument('--device', type=str, help='cpu/cuda', default='cpu')
    parser.add_argument('--num_workers', type=int, help='number of workers', default=12)
    parser.add_argument('--skip_irt', action='store_true', help='skip irt')
    parser.add_argument('--cache_path', type=str, help='cache path', default=None)
    parser.add_argument('--sampling_names', type=str, help='sampling names', default='random,anchor,anchor-irt')
    parser.add_argument('--filename_suffix', type=str, help='path suffix', default='')
    parser.add_argument('--make_results_table', action='store_true', help='make results table')

    apply_random_seed(RANDOM_SEED)

    args = parser.parse_args()
    bench = args.bench
    split = args.split
    iterations = args.iterations
    device = args.device

    assert bench in ['helm_lite','lb','mmlu','alpaca','mmlu_fields','icl_templates']
    assert split in ['iid','noniid','noniid2','noniid3']
    assert iterations>0

    # Defining other parameters
    Ds = [2, 5, 10, 15]
    sampling_names = args.sampling_names.split(',')

    scenario_name = 'full' #we are evaluating all scenarios at once (this is just a nomination)

    data, scenarios, set_of_rows = get_data(bench, split)

    chosen_scenarios = list(scenarios.keys())

    if args.cache_path is not None:
        if os.path.exists(args.cache_path):
            cache = load_pickle(args.cache_path)
        else:
            dirname = os.path.dirname(args.cache_path)
            if dirname != '':
                os.makedirs(dirname, exist_ok=True)
            cache = {"cache_path": args.cache_path}
    else:
        cache = None

    # ## Results
    results_full, accs_full, sampling_time_dic = evaluate_scenarios(
        data,
        scenario_name,
        chosen_scenarios,
        scenarios,
        set_of_rows,
        Ds,
        iterations,
        device,
        bench='irt_'+bench,
        split=split,
        sampling_names=sampling_names,
        num_workers=args.num_workers,
        skip_irt=args.skip_irt,
        cache=cache
    )

    if args.cache_path is not None:
        dump_pickle(cache, args.cache_path)

    filename_suffix = args.filename_suffix

    results_full_path = f'results/results_{bench}_split-{split}_iterations-{iterations}{filename_suffix}.pickle'
    accs_full_path = f'results/accs_{bench}_split-{split}_iterations-{iterations}{filename_suffix}.pickle'
    samplingtime_full_path = f'results/samplingtime_{bench}_split-{split}_iterations-{iterations}{filename_suffix}.pickle'

    with open(results_full_path, 'wb') as handle:
        pickle.dump(results_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(accs_full_path, 'wb') as handle:
        pickle.dump(accs_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(samplingtime_full_path, 'wb') as handle:
        pickle.dump(sampling_time_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.make_results_table:
        table_avg, table_std = make_table_avg(
            bench,
            split,
            filename_suffix,
            accs_full,
            scenarios_to_skip=SCENARIOS_TO_SKIP
        )
        make_results_table(table_avg, table_std, bench)


def make_table_avg(bench, split, filename_suffix, accs_full, scenarios_to_skip):
    table_avg = {}
    table_std = {}
    # model_perf = {} # not used?
    # for bench in benchs:
    table_avg[bench] = {}
    table_std[bench] = {}
    # model_perf[bench] = {}

    agg = 'leaderboard' # 'leaderboard', 'scenarios'
    results = 'acc'# 'acc', 'rank'

    if results == 'acc': ylim = (0,.1)
    elif results == 'rank':
        if agg_metric == 'std': ylim = (0,.1)
        else: ylim = (.5,1)
    else: raise NotImplementedError

    # for split in splits[bench]:
    table_avg[bench][split] = {}
    table_std[bench][split] = {}
    # model_perf[bench][split] = {}

    if bench == 'mmlu_fields' and split == 'iid':
        filename_suffix = filename_suffix
    else:
        filename_suffix = ''

    # full_results_path = f'results/accs_{bench}_split-{split}_iterations-5{filename_suffix}.pickle'

    # with open(full_results_path, 'rb') as handle:
    #     data = pickle.load(handle)

    data = accs_full

    models = list(data.keys())
    number_items = list(data[models[0]].keys())
    methods = list(data[models[0]][number_items[0]].keys())
    scenarios = list(data[models[0]][number_items[0]][methods[0]].keys())

    data = np.array([[[[data[model][number_item][method][scenario] for scenario in scenarios]  for model in data.keys()] for number_item in number_items] for method in methods])
    scores = load_scores(bench, split, scenarios_to_skip=scenarios_to_skip)

    if agg == 'leaderboard':
        if bench=='helm':
            ###
            if results == 'acc':
                data = np.abs(winrate(data, axis=2).mean(axis=3)-winrate(scores, axis=1).mean(axis=0)[None,None,:,None])
            elif results == 'rank':
                rank_corrs = np.zeros(data.mean(axis=2).mean(axis=2).shape)
                #print(bench,rank_corrs.shape)
                for i in range(rank_corrs.shape[0]):
                    for j in range(rank_corrs.shape[1]):
                        for l in range(rank_corrs.shape[2]):
                            #print(winrate(data, axis=2).mean(axis=3).shape)
                            rank_corrs[i,j,l] = stats.spearmanr(winrate(data, axis=2).mean(axis=3)[i,j,:,l], winrate(scores.T, axis=0).mean(axis=1)).statistic
                data=rank_corrs

            else:
                raise NotImplementedError
        else:
            ###
            if results == 'acc':

                data = np.abs(data.mean(axis=3)-scores.mean(axis=0)[None,None,:,None])
            elif results == 'rank':
                rank_corrs = np.zeros(data.mean(axis=2).mean(axis=2).shape)
                #print(bench,rank_corrs.shape)
                for i in range(rank_corrs.shape[0]):
                    for j in range(rank_corrs.shape[1]):
                        for l in range(rank_corrs.shape[2]):
                            #print(data.mean(axis=3).shape)
                            rank_corrs[i,j,l] = stats.spearmanr(data.mean(axis=3)[i,j,:,l], scores.T.mean(axis=1)).statistic
                data=rank_corrs
            else:
                raise NotImplementedError
    elif agg == 'scenarios':
        if results == 'acc':
            data = np.abs(data-scores.T[None,None,:,:,None]).mean(axis=3)
        elif results == 'rank':
            rank_corrs = np.zeros(data.mean(axis=2).shape)
            for i in range(rank_corrs.shape[0]):
                for j in range(rank_corrs.shape[1]):
                    for k in range(rank_corrs.shape[2]):
                        for l in range(rank_corrs.shape[3]):
                            rank_corrs[i,j,k,l] = stats.spearmanr(data[i,j,:,k,l], scores.T[:,k]).statistic
            data=rank_corrs
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if agg_metric=='avg':
        data = data.mean(-1) #iterations
    elif agg_metric=='std':
        data = data.std(-1)
    else:
        raise NotImplementedError


    for i,method in enumerate(methods):
        table_avg[bench][split][method] = {}
        table_std[bench][split][method] = {}

        for j,number_item in enumerate(number_items):
            if agg == 'leaderboard' and results == 'rank':
                #print(data.shape)
                table_avg[bench][split][method][number_item] = data[i,j]
                table_std[bench][split][method][number_item] = 0
            else:
                #print(data.shape)
                table_avg[bench][split][method][number_item] = np.mean(data, axis=-1)[i,j]
                table_std[bench][split][method][number_item] = data.std(-1)[i,j]

    return table_avg, table_std


def make_results_table(table_avg, table_std, bench):

    agg = 'leaderboard' # 'leaderboard', 'scenarios'
    results = 'acc'# 'acc', 'rank'

    style = {"alpha":1, "markersize":3, "markeredgewidth":1, "elinewidth":1, "capsize":3, "linestyle":''}

    # Load table_avg from pickle file
    with open('results/table_avg_original.pickle', 'rb') as handle:
        table_avg_original = pickle.load(handle)
    with open('results/table_std_original.pickle', 'rb') as handle:
        table_std_original = pickle.load(handle)

    table_avg = merge_methods(table_avg, table_avg_original)
    table_std = merge_methods(table_std, table_std_original)

    if results == 'acc': ylim = (0,.1)
    elif results == 'rank':
        if agg_metric == 'std': ylim = (0,.1)
        else: ylim = (.5,1)
    else: raise NotImplementedError


    cur_methods = table_avg["mmlu_fields"]["iid"].keys()
    cur_methods = [method for method in cur_methods if method not in ['random_pirt', 'anchor_pirt', 'random_cirt', 'anchor_cirt', "anchor-irt_cirt", "anchor-irt_pirt"]]
    cur_methods = [method for method in cur_methods if method not in ["anchor_gpirt", "random_gpirt", "anchor_naive", "random_naive", "anchor-irt_naive"]]

    cur_methods_for_table = table_avg["mmlu_fields"]["iid"].keys()

    # Iterate over your benchmarks
    for i, split in enumerate(splits[bench]):  # Replace `benchmarks` with your list of benchmarks

        if split == 'noniid':
            print("DEBUG: skipping noniid for now")
            continue

        df = make_perf_table(
            table_avg[bench][split],
            table_std[bench][split],
            methods=cur_methods_for_table,
        )

        pd.set_option('display.max_rows', MAX_TABLE_SIZE)
        pd.set_option('display.max_columns', MAX_TABLE_SIZE)
        pd.set_option(
            "display.max_colwidth", MAX_TABLE_SIZE
        )
        for num_samples in df.keys():
            print("#anchor_points:", num_samples)
            # Reorder columns to put guiding models, PDS type, and stratified first
            cols = df[num_samples].columns.tolist()
            first_cols = ['#guiding_models', 'PDS type', 'stratified']
            other_cols = [col for col in cols if col not in first_cols]
            df[num_samples] = df[num_samples][first_cols + other_cols]

            # Replace all values in #guiding_models column with 382
            df[num_samples].loc[df[num_samples]['#guiding_models'] == 'all', '#guiding_models'] = 382

            # Sort rows by #guiding_models
            df[num_samples] = df[num_samples].sort_values(['PDS type', 'stratified', '#guiding_models'])

            print(df[num_samples])

    df[max(list(df.keys()))].to_csv('df_100.csv')

if __name__ == "__main__":
    main()
