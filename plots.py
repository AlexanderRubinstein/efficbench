import pickle
import copy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#from experiments import *
from utils import *

# Define the new renamings according to the latest instructions
rename_mappings = {
    'random_naive': 'random',
    'anchor_naive': 'correct.',
    'anchor-irt_naive': 'IRT',
    'random_pirt': 'random+',
    'anchor_pirt': 'correct.+',
    'anchor-irt_pirt': 'IRT+',
    'random_gpirt': 'random++',
    'anchor_gpirt': 'correct.++',
    'anchor-irt_gpirt': 'IRT++',
    'anchor': 'correct.',
    'anchor-irt':'IRT',
    # "high-disagreement_naive": "PDS",
    "mean_train_score": "mean train score" # [ADD][new estimator]
}

color_mappings = {
    'random_naive': '#8c564b',
    'anchor_naive': '#1f77b4',
    'anchor-irt_naive': '#2ca02c',
    'random_gpirt': '#9467bd',
    'anchor_gpirt': '#d62728',
    'anchor-irt_gpirt': '#ff7f0e',
    'anchor': '#1f77b4',
    'anchor-irt': '#2ca02c',
    # "high-disagreement_naive": "#9467bd",
    "mean_train_score": "#000000", # [ADD][new estimator]
    #
    # 'random_pirt': '#1f77b4',
    # 'anchor_pirt': '#1f77b4',
    # 'anchor-irt_pirt': '#2ca02c',
}

CONSTANT_ESTIMATORS = [
    "mean_train_score",
    "perfect_knn",
]

EXTRA_COLORS = [
    '#17becf',
    '#bcbd22',
    '#7f7f7f',
    '#e377c2',
    '#ffbb78',
    '#98df8a',
    '#ff9896',
    '#c5b0d5',
    '#c49c94',
    '#f7b6d2',
    '#9467bd',
    '#8c564b',
    '#1f77b4',
    '#2ca02c',
]

benchs = ['lb', 'mmlu', 'helm_lite', 'alpaca','mmlu_fields', 'icl_templates']
titles = {'lb':'Open LLM Leaderboard','mmlu':'MMLU','helm_lite':'HELM','alpaca':'AlpacaEval', 'icl_templates':'ICL consistency'}
splits = {'lb':['iid','noniid'],'mmlu':['iid','noniid'],
          'helm_lite':['iid','noniid'],'alpaca':['iid','noniid'],
          'mmlu_fields':['iid','noniid'],'icl_templates':['iid','noniid','noniid2','noniid3']}

agg_metric = 'avg' #'std' (std=variation across seeds)
methods = ['random_naive', 'anchor_naive', 'anchor-irt_naive',
           #'random_pirt', 'anchor_pirt', 'anchor-irt_pirt']#,
           #'random_cirt','anchor_cirt', 'anchor-irt_cirt']#,
           'random_gpirt', 'anchor_gpirt', 'anchor-irt_gpirt']


style = {"alpha":1, "markersize":3, "markeredgewidth":1, "elinewidth":1, "capsize":3, "linestyle":''}

def plot_perf_lines(table_avg, table_std, title, xlabel, ylabel, ylim,
                    legend=False, error_bar=False, show_title=True, show_xlabel=True, show_ylabel=True, ncols=6, posic=(-1.5, -.35)):

    markers = ['o', 'v', '*', 'x', 's','p']
    jitters = [-6.3,-3.7,-1.3,1.3,3.7,6.3]
    colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][:9]
    j = 0
    for method, values in table_avg.items():
        x = np.array(list(values.keys()))
        y = np.array(list(values.values()))
        s = np.array(list(table_std[method].values()))

        if error_bar:
            plt.errorbar(
                (x + jitters[j]),
                y,
                color=color_mappings[method],
                yerr=s,
                # label=rename_mappings.get(method, method),
                label=rename_mappings[method],
                marker=markers[j],
                **style
            )
        else:
            plt.plot(x, y, label=method)

        j+=1
    if show_xlabel: plt.xlabel(xlabel, size=11)
    if show_ylabel: plt.ylabel(ylabel, size=11)
    plt.ylim(ylim[0], ylim[1])
    if show_title:
        plt.title(title)
    else:
        pass

    tick_label_size = 10  # Example size, adjust as needed
    plt.tick_params(axis='x', labelsize=tick_label_size)
    plt.tick_params(axis='y', labelsize=tick_label_size)

    if legend: plt.legend(loc='upper center', ncols=ncols, bbox_to_anchor=posic)
    plt.grid(alpha=.2)
    #plt.grid(which='major', color='black', linestyle='-')
    #plt.grid(which='minor', color='gray', linestyle=':')
    #plt.show()


def plot_perf_lines_v2(
    table_avg,
    table_std,
    methods,
    title,
    xlabel,
    ylabel,
    ylim,
    legend=False,
    error_bar=False,
    show_title=True,
    show_xlabel=True,
    show_ylabel=True,
    ncols=6,
    posic=(-1.5, -.35)
):

    # markers = ['o', 'v', '*', 'x', 's','p']
    # jitters = [-6.3,-3.7,-1.3,1.3,3.7,6.3]
    total_methods = len(methods)

    markers = ['o', 'v', '*', 'x', 's','p'] + ['o'] * total_methods
    # jitters = [-6.3,-3.7,-1.3,1.3,3.7,6.3]
    jitters = np.linspace(-6.3, 6.3, total_methods)
    colors = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][:9]
    j = 0
    extra_color_idx = 0
    extra_colors = EXTRA_COLORS
    # for method, values in table_avg.items():

    for method in methods:

        cur_color = color_mappings.get(method)
        if cur_color is None:
            cur_color = extra_colors[extra_color_idx]
            extra_color_idx += 1

        if method == 'mean_train_score':
            values = table_avg[method]
            x = np.array(list(values.keys()))
            y = np.array(list(values.values()))
            s = np.array(list(table_std[method].values()))

            # Plot horizontal lines for mean and mean ± std
            plt.axhline(y=np.mean(y), color=cur_color, linestyle='--', alpha=0.5, label=rename_mappings.get(method, method))
            # plt.axhline(y=np.mean(y) + np.mean(s), color=cur_color, linestyle=':', alpha=0.3)
            # plt.axhline(y=np.mean(y) - np.mean(s), color=cur_color, linestyle=':', alpha=0.3)
            continue

        values = table_avg[method]
        x = np.array(list(values.keys()))
        y = np.array(list(values.values()))
        s = np.array(list(table_std[method].values()))

        if error_bar:
            plt.errorbar(
                (x + jitters[j]),
                y,
                color=cur_color,
                yerr=s,
                label=rename_mappings.get(method, method),
                marker=markers[j],
                **style
            )
        else:
            plt.plot(x, y, label=method)

        j+=1
    if show_xlabel: plt.xlabel(xlabel, size=11)
    if show_ylabel: plt.ylabel(ylabel, size=11)
    plt.ylim(ylim[0], ylim[1])
    if show_title:
        plt.title(title)
    else:
        pass

    tick_label_size = 10  # Example size, adjust as needed
    plt.tick_params(axis='x', labelsize=tick_label_size)
    plt.tick_params(axis='y', labelsize=tick_label_size)

    if legend: plt.legend(loc='upper center', ncols=ncols, bbox_to_anchor=posic)
    plt.grid(alpha=.2)
    #plt.grid(which='major', color='black', linestyle='-')
    #plt.grid(which='minor', color='gray', linestyle=':')
    #plt.show()


def make_perf_table(
    table_avg,
    table_std,
    methods,
):

    df_dict = {}

    for method in methods:

        for number_item in table_avg[method].keys():
            if number_item not in df_dict.keys():
                df_dict[number_item] = {}
            dict_per_num_anchors = df_dict[number_item]

            if method in CONSTANT_ESTIMATORS:
                sampling_name = "-"
                prediction_name = method
            else:
                split = method.split('_')
                sampling_name, prediction_name = split[0], "_".join(split[1:])

            if sampling_name not in dict_per_num_anchors.keys():
                dict_per_num_anchors[sampling_name] = {}

            values = table_avg[method]
            estimation_error = values[number_item]

            dict_per_num_anchors[sampling_name][prediction_name] = estimation_error

            if "disagreement" in sampling_name:
                if "@" in sampling_name:
                    n_guiding_models = int(sampling_name.split("@")[1].split("+")[0])
                    dict_per_num_anchors[sampling_name]["#guiding_models"] = n_guiding_models
                else:
                    dict_per_num_anchors[sampling_name]["#guiding_models"] = "all"
                if "high" in sampling_name:
                    dict_per_num_anchors[sampling_name]["PDS type"] = "highest"
                else:
                    dict_per_num_anchors[sampling_name]["PDS type"] = "lowest"
                if "+nonstratified" in sampling_name:
                    dict_per_num_anchors[sampling_name]["stratified"] = False
                else:
                    dict_per_num_anchors[sampling_name]["stratified"] = True

    return {num_samples: pd.DataFrame(df_dict[num_samples]).T for num_samples in df_dict.keys()}


def winrate(x,axis):
    n = x.shape[axis]
    return(np.argsort(np.argsort(x, axis=axis), axis=axis)/n)

def load_scores(bench, split, scenarios_to_skip=[], ordered=False):
    with open(f'results/accs_{bench}_split-{split}_iterations-5.pickle', 'rb') as handle:
        data = pickle.load(handle)

    if bench=='mmlu':
        with open(f'data/lb.pickle', 'rb') as handle:
            data2 = pickle.load(handle)
    elif bench=='alpaca':
        with open(f'data/alpaca_v2.pickle', 'rb') as handle:
            data2 = pickle.load(handle)
    else:
        data2_path = f'data/{bench}.pickle' if not ordered else f'data/{bench}_ordered.pickle'
        with open(data2_path, 'rb') as handle:
            data2 = pickle.load(handle)
    if bench=='lb':scenarios = lb_scenarios
    elif bench=='mmlu':scenarios = {'mmlu':lb_scenarios['mmlu']}
    elif bench=='helm_lite':scenarios = helm_lite_scenarios
    elif bench=='alpaca':scenarios = alpaca_scenarios
    elif bench=='mmlu_fields':scenarios = {'mmlu':lb_scenarios['mmlu']}
    elif bench=='icl_templates':scenarios = icl_templates_scenarios
    else: raise NotImplementedError

    filtered_scenarios = {}
    scenarios_to_pop = []
    for scenario in scenarios:
        filtered_scenarios[scenario] = []
        for sub in scenarios[scenario]:
            if sub in scenarios_to_skip:
                # print(bench, split)
                print(f"Sub-scenario {sub} of scenario {scenario} is in the skip list: {scenarios_to_skip}, so skipping it")
                continue
            if sub not in data2['data']:
                print(f"Sub-scenario {sub} of scenario {scenario} not found in data2, so skipping it")
                continue
            filtered_scenarios[scenario].append(sub)
        if len(filtered_scenarios[scenario]) == 0:
            scenarios_to_pop.append(scenario)
    for scenario in scenarios_to_pop:
        del filtered_scenarios[scenario]
    scenarios = filtered_scenarios
    chosen_scenarios = list(filtered_scenarios.keys())

    scenarios_position, subscenarios_position = prepare_data(chosen_scenarios, scenarios, data2)
    scores = create_responses(chosen_scenarios, scenarios, data2)

    # Balance weights
    balance_weights = np.ones(scores.shape[1])
    for scenario in scenarios:
        N = len(scenarios_position[scenario])
        n_sub = len(scenarios[scenario])
        for sub in scenarios[scenario]:
            n_i = len(subscenarios_position[scenario][sub])
            balance_weights[subscenarios_position[scenario][sub]] = N/(n_sub*n_i)

    scores = balance_weights*scores

    scores = np.vstack([scores[:,scenarios_position[scenario]].mean(axis=1) for scenario in scenarios])

    return scores[:,list(data.keys())]


def make_table_avg(
    bench,
    split,
    filename_suffix,
    accs_full,
    scenarios_to_skip,
    return_perf_table=False,
    ordered=True,
    results='acc'
):
    table_avg = {}
    table_std = {}
    model_perf = {} # not used?
    # for bench in benchs:
    table_avg[bench] = {}
    table_std[bench] = {}
    model_perf[bench] = {}

    agg = 'leaderboard' # 'leaderboard', 'scenarios'
    # results = 'acc'# 'acc', 'rank'

    if results == 'acc': ylim = (0,.1)
    elif results == 'rank':
        if agg_metric == 'std': ylim = (0,.1)
        else: ylim = (.5,1)
    else: raise NotImplementedError

    # for split in splits[bench]:
    table_avg[bench][split] = {}
    table_std[bench][split] = {}
    model_perf[bench][split] = {}

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
    scores = load_scores(bench, split, scenarios_to_skip=scenarios_to_skip, ordered=ordered)

    if agg == 'leaderboard':
        if bench=='helm':
            ###
            if results == 'acc':
                ###
                model_perf[bench][split]['truth'] = winrate(scores, axis=1).mean(axis=0)
                for i,method in enumerate(methods):
                    model_perf[bench][split][method] = {}
                    model_perf[bench][split][method] = {}
                for j,number_item in enumerate(number_items):
                    model_perf[bench][split][method][number_item] = winrate(data, axis=2).mean(axis=3)[i,j,:,:]
                ###
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

                # ###
                model_perf[bench][split]['truth'] = scores.mean(axis=0)
                for i,method in enumerate(methods):
                    model_perf[bench][split][method] = {}
                    model_perf[bench][split][method] = {}
                    for j,number_item in enumerate(number_items):
                        model_perf[bench][split][method][number_item] = data.mean(axis=3)[i,j,:,:]
                # ###

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

    res = [table_avg, table_std]
    if return_perf_table:
        res = res + [model_perf]
    return res
