from tqdm import tqdm
import pickle
from copy import copy
import multiprocessing as mp
import time
from irt import *
from selection import *
from utils import load_pickle, dump_pickle
from acc import *


def make_cache_key(scenario_name, split_number, suffix):
    return f'{scenario_name}_{split_number}_{suffix}'


def evaluate_scenarios(
    data,
    scenario_name,
    chosen_scenarios,
    scenarios,
    set_of_rows,
    Ds,
    iterations,
    device,
    bench,
    split,
    sampling_names=['random', 'anchor', 'anchor-irt'],
    num_workers=1,
    skip_irt=False,
    cache=None
):


    """
    Evaluates scenarios by training and validating IRT models, then computing accuracies and updating results.

    Parameters:
    - data: A dictionary containing the dataset.
    - scenario_name: The name of the current scenario.
    - chosen_scenarios: A list of scenarios to be considered.
    - scenarios: A dictionary mapping each scenario to its subscenarios.
    - set_of_rows: A set of row indices to hide during training (to simulate missing data).
    - Ds: A list of dimension values to consider for the IRT model.
    - iterations: The number of iterations to perform for random evaluations.
    - device: The computing device ('cpu' or 'gpu') to use for training.

    Returns:
    - A dictionary containing the updated results.
    """

    assert bench in ['irt_helm_lite', 'irt_lb', 'irt_lb_perf', 'irt_mmlu', 'irt_alpaca', 'irt_mmlu_fields', 'irt_icl_templates']
    assert np.mean([s in ['random', 'anchor', 'anchor-irt', 'adaptive'] for s in sampling_names]) == 1

    number_items = [10, 30, 60, 100]  # Number of items to consider in evaluations

    # cpu = mp.cpu_count()  # Number of available CPU cores
    cpu = num_workers
    epochs = 2000  # Number of epochs for IRT model training (package default is 2000)
    lr = .1  # Learning rate for IRT model training (package default is .1)

    # Iterate through each set of rows to hide
    accs_true = {}  # Initialize a dictionary to hold real accuracies
    out = [] # To store intermediate results
    for split_number, rows_to_hide in enumerate(set_of_rows):
        rows_to_hide_str = ':'.join([str(r) for r in rows_to_hide])[:30] + ':'.join([str(r) for r in rows_to_hide])[-30:]

        print(f"\nEvaluating models {rows_to_hide}")

        # Prepare data and scenarios
        scores_train, scores_test, balance_weights, scenarios_position, subscenarios_position = prepare_and_split_data(chosen_scenarios, scenarios, data, rows_to_hide)

        responses_train = np.zeros(scores_train.shape)
        responses_test = np.zeros(scores_test.shape)

        # Threshold responses
        cs = np.linspace(0.01,.99,1000)  # Threshold values to consider
        for scenario in chosen_scenarios:

            ind = scenarios_position[scenario]

            if cache is not None:
                cache_key = make_cache_key(scenario_name, split_number, f'{scenario}_{chosen_scenarios}_c')
            else:
                cache_key = None
            if cache_key is not None and cache_key in cache:
                c = cache[cache_key]
            else:
                # Find the best threshold value that minimizes the difference between mean responses and mean scores
                c = cs[np.argmin([np.mean((np.abs((scores_train[:,ind]>c).mean(axis=1)-scores_train[:,ind].mean(axis=1)))) for c in cs])]
                # Apply the threshold to train and test responses
                if cache_key is not None:
                    cache[cache_key] = c
                    dump_pickle(cache, cache["cache_path"])
            responses_train[:,ind] = (scores_train[:,ind]>c).astype(int)
            responses_test[:,ind] = (scores_test[:,ind]>c).astype(int)

        # Storing true accs to use later
        for j in range(len(rows_to_hide)):
            accs_true[rows_to_hide[j]] = {}
            for scenario in chosen_scenarios:
                accs_true[rows_to_hide[j]][scenario] = ((balance_weights[None,:]*scores_test)[j, scenarios_position[scenario]]).mean()

        if skip_irt:
            A, B, Theta, opt_lambds = None, None, None, None
        else:

            # Choosing D through validation
            val_ind = list(range(0,responses_train.shape[0],5)) #list(range(int(responses_train.shape[0]/3)))
            train_ind = [i for i in range(responses_train.shape[0]) if i not in val_ind]

            errors = []  # Initialize a list to hold validation errors
            errors2 = []

            if cache is not None:
                cache_key = make_cache_key(scenario_name, split_number, 'D')

            if cache_key is not None and cache_key in cache:
                ind_D, D, errors2 = cache[cache_key]
            else:
                print("\ni) choosing optimal D")
                for D in tqdm(Ds):
                    # Train IRT model for the current dimension (D)
                    model_name = f'models/{bench}/split-{split}{split_number}_D-{D}_scenario-{scenario_name}_val/'
                    # Load trained IRT model parameters
                    try:
                        A, B, Theta = load_irt_parameters(model_name)
                    except:
                        # Create IRT dataset for validation and train IRT models
                        dataset_name = f'data/{bench}/split-{split}{split_number}_scenario-{scenario_name}_val.jsonlines'
                        create_irt_dataset(responses_train[train_ind], dataset_name)
                        train_irt_model(dataset_name, model_name, D, lr, epochs, device)
                        A, B, Theta = load_irt_parameters(model_name)
                    # Determine seen and unseen items for validation
                    seen_items = list(range(0, responses_train.shape[1], 2))
                    unseen_items = list(range(1, responses_train.shape[1], 2))
                    # Estimate ability parameters for the validation set
                    print(" - fit. theta in the val set")
                    pool = mp.Pool(cpu)
                    thetas = pool.starmap(estimate_ability_parameters, [(responses_train[val_ind][j], seen_items, A, B) for j in range(len(val_ind))])
                    pool.close()
                    pool.join()

                    # Compute validation errors for each scenario and update the errors list (in the end, we give the same weight for all scenarios)
                    errors2.append([])
                    for scenario in chosen_scenarios:
                        ind = [u for u in unseen_items if u in scenarios_position[scenario]]
                        errors2[-1].append(np.mean([abs((balance_weights*item_curve(thetas[j], A, B))[0,ind].mean()-scores_train[val_ind][j,ind].mean())for j in range(len(val_ind))]))
                    errors.append(np.mean(errors2[-1]))
                    print(errors[-1])

                # Choose the simplest model (D) that is not far from the best model based on validation errors
                ind_D = np.argmax(np.array(errors)-np.min(errors)<.0025)
                D = Ds[ind_D]
                print("- opt D=", D, "errors=", errors, "\n")

                if cache_key is not None:
                    cache[cache_key] = ind_D, D, errors2
                    dump_pickle(cache, cache["cache_path"])

            # Choosing lambdas (For random G-PIRT)
            print("\nii) choosing optimal lambdas")

            opt_lambds = {'random_gpirt': {}, 'anchor_gpirt': {}, 'anchor-irt_gpirt': {}, 'adaptive_gpirt': {}}  # Initialize a dictionary to hold optimal lambda values

            vs = {}
            bs = {}
            for i,scenario in enumerate(chosen_scenarios):
                vs[scenario] = np.var(scores_train[:,scenarios_position[scenario]], axis=1).mean()
                bs[scenario] = np.mean(errors2[ind_D][i])

            for scenario in tqdm(chosen_scenarios):
                for key in opt_lambds.keys():
                    opt_lambds[key][scenario] = {}
                    for number_item in number_items:
                        if key == 'random_gpirt':
                            opt_lambds[key][scenario][number_item] = get_lambda(bs[scenario], vs[scenario]/number_item)
                        else:
                            opt_lambds[key][scenario][number_item] = get_lambda(bs[scenario], vs[scenario]/(4*number_item))

            # Save the final dataset and train the final IRT model
            print("\niii) fitting final IRT model")

            model_name = f'models/{bench}/split-{split}{split_number}_D-validate_scenario-{scenario_name}/'
            # Load the final IRT model
            try:
                A, B, Theta = load_irt_parameters(model_name)
            except:
                dataset_name = f'data/{bench}/split-{split}{split_number}_scenario-{scenario_name}.jsonlines'
                create_irt_dataset(responses_train, dataset_name)
                train_irt_model(dataset_name, model_name, D, lr, epochs, device)
                A, B, Theta = load_irt_parameters(model_name)

        print("\niv) sampling")
        item_weights_dic, seen_items_dic, unseen_items_dic, sampling_time_dic = {}, {}, {}, {}
        if cache is not None:
            cache_key = make_cache_key(scenario_name, split_number, 'sampling')
        else:
            cache_key = None
        for sampling_name in tqdm(sampling_names):
            if skip_irt and sampling_name == 'anchor-irt':
                continue

            if 'adaptive' in sampling_name:
                raise NotImplementedError
                item_weights_dic[sampling_name] = {number_item: {n_model: [] for n_model in range(responses_test.shape[0])} for number_item in number_items}
                seen_items_dic[sampling_name] = {number_item: {n_model: [] for n_model in range(responses_test.shape[0])} for number_item in number_items}
                unseen_items_dic[sampling_name] = {number_item: {n_model: [] for n_model in range(responses_test.shape[0])} for number_item in number_items}
                sampling_time_dic[sampling_name] = {number_item: [] for number_item in number_items}

                inital_items = select_initial_adaptive_items(A, B, Theta, D+1, try_size=10000)

                pool = mp.Pool(cpu)
                # parallelising models
                samples = pool.starmap(sample_items_adaptive, [(number_items, iterations, sampling_name, chosen_scenarios, scenarios,
                                                                subscenarios_position, responses, scores_train, scenarios_position,
                                                                A, B, balance_weights, inital_items) for responses in responses_test])
                pool.close()
                pool.join()
                for n_model, samples_model in enumerate(samples): #zip(rows_to_hide, samples):
                    item_weights_model, seen_items_model, unseen_items_model, sampling_time = samples_model
                    sampling_time = np.array(sampling_time).mean()
                    for number_item in number_items:
                        item_weights_dic[sampling_name][number_item][n_model] = item_weights_model[number_item]
                        seen_items_dic[sampling_name][number_item][n_model] = seen_items_model[number_item]
                        unseen_items_dic[sampling_name][number_item][n_model] = unseen_items_model[number_item]
                        sampling_time_dic[sampling_name][number_item].append(sampling_time)
            else:

                if cache_key is not None and cache_key in cache:
                    # samples, item_weights_dic, seen_items_dic, unseen_items_dic, sampling_time_dic = cache[cache_key]
                    item_weights_dic, seen_items_dic, unseen_items_dic, sampling_time_dic = cache[cache_key]
                else:
                    # parallelising number of items
                    pool = mp.Pool(cpu)
                    item_weights_dic[sampling_name], seen_items_dic[sampling_name], unseen_items_dic[sampling_name], sampling_time_dic[sampling_name] = {}, {}, {}, {}
                    samples = pool.starmap(
                        sample_items,
                        [(
                            number_item,
                            iterations,
                            sampling_name,
                            chosen_scenarios,
                            scenarios,
                            subscenarios_position,
                            responses_test,
                            scores_train,
                            scenarios_position,
                            A,
                            B,
                            balance_weights,
                            skip_irt
                        ) for number_item in number_items]
                    )
                    pool.close()
                    pool.join()

                    for i, number_item in enumerate(number_items):
                        (
                            item_weights_dic[sampling_name][number_item],
                            seen_items_dic[sampling_name][number_item],
                            unseen_items_dic[sampling_name][number_item],
                            sampling_time_dic[sampling_name][number_item]
                        ) = samples[i]

        if cache_key is not None:
            cache[cache_key] = (
                # samples,
                item_weights_dic,
                seen_items_dic,
                unseen_items_dic,
                sampling_time_dic
            )
            dump_pickle(cache, cache["cache_path"])

        #saving points
        if rows_to_hide==set_of_rows[0] and split=='iid': # we impose rows_to_hide==set_of_rows[0] because in AlpacaEval we do K-fold CV so there are multiple training/test sets
            dic = {}
            dic['item_weights'] = item_weights_dic
            dic['seen_items'] = seen_items_dic
            dic['scenarios_position'] = scenarios_position
            dic['subscenarios_position'] = subscenarios_position
            dic['opt_lambds'] = opt_lambds
            dic['A'] = A
            dic['B'] = B
            with open(f'results/samples_{bench[4:]}_iterations-{iterations}.pickle', 'wb') as handle:
                pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print("\nv) computing accuracies")
        start_time = time.time()

        pool = mp.Pool(cpu)
        out += pool.starmap(
            calculate_accuracies,
            [
                (
                    j,
                    sampling_names,
                    item_weights_dic,
                    seen_items_dic,
                    unseen_items_dic,
                    A,
                    B,
                    scores_test,
                    scores_train,
                    responses_test,
                    scenarios_position,
                    chosen_scenarios,
                    balance_weights,
                    opt_lambds,
                    rows_to_hide,
                    skip_irt
                ) for j in tqdm(range(len(rows_to_hide)))
            ]
        )
        pool.close()
        pool.join()
        elapsed_time = np.round(time.time()-start_time)
        print(f" - finished in {elapsed_time} seconds")

    ### Final results
    accs_hat = {}
    results = {}
    for item in out:
        key = list(item.keys())[0]
        accs_hat[key] = item[key]

    # Update results with the mean absolute difference for each approach
    for rows_to_hide in set_of_rows:
        for j in range(len(rows_to_hide)):
            results[rows_to_hide[j]] = {}
            for number_item in number_items:
                results[rows_to_hide[j]][number_item] = {}
                for sampling_name in sampling_names:
                    if skip_irt and sampling_name == 'anchor-irt':
                        continue
                    # for estimators in ['naive', 'pirt', 'cirt', 'gpirt']:
                    for estimators in ESTIMATORS:
                        if skip_irt and estimators in ['pirt', 'cirt', 'gpirt']:
                            continue
                        method_name = make_method_name(sampling_name, estimators)
                        results[rows_to_hide[j]][number_item][method_name] = {}
                        for scenario in chosen_scenarios:
                            acc_hat = accs_hat[rows_to_hide[j]][number_item][
                                method_name
                            ][scenario]
                            acc_true = accs_true[rows_to_hide[j]][scenario]
                            results[rows_to_hide[j]][number_item][method_name][
                                scenario
                            ] = np.abs(np.array(acc_hat) - acc_true)

    return results, accs_hat, sampling_time_dic # Return the updated results dictionary
