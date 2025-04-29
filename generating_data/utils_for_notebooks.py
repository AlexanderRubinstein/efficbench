from tqdm import tqdm
from stnd.utility.utils import parse_list_from_string
import numpy as np


MAX_NUM_ANSWERS = 31 # 5 for arc_harness_25, 31 for truth_harness_mc
KEYS_TO_ADD = ["correctness", "predictions"]
SUB_TO_SKIP = ["harness_gsm8k_5"]


def pad_predictions(predictions, max_num_answers=MAX_NUM_ANSWERS):
    if len(predictions) < max_num_answers:
        predictions.extend([-float('inf')] * (max_num_answers - len(predictions)))
    return predictions


def parse_df_with_results(
    df,
    models,
    order,
    sub_to_skip=SUB_TO_SKIP,
    max_num_answers=MAX_NUM_ANSWERS,
    keys_to_add=KEYS_TO_ADD
):

    if order is None:
        models = models
    else:
        models = [models[o] for o in order]

    data = {}
    data['data'] = {}
    data['models'] = models
    max_answers_dict = {}

    for sub in tqdm(df[list(df.keys())[0]].keys()):
        if sub in sub_to_skip:
            continue
        max_answers_dict[sub] = 0
        data['data'][sub] = {}
        # data['data'][sub]['correctness'] = []

        for key in keys_to_add:
            data['data'][sub][key] = []

        for model in models:
            # data['data'][sub]['correctness'].append(df[model][sub]['correctness'])
            for key in keys_to_add:
                if key not in df[model][sub].keys():
                    value_to_add = None
                else:
                    value_to_add = df[model][sub][key]
                if key == 'predictions':
                    new_value_to_add = []
                    for model_preds in value_to_add:

                        if isinstance(model_preds, str):
                            parsed_model_preds = parse_list_from_string(model_preds, list_separators=[','])
                            new_value_to_add.append(pad_predictions(parsed_model_preds))

                        else:
                            new_value_to_add.append(pad_predictions(model_preds))

                        assert len(new_value_to_add[-1]) == max_num_answers, \
                            f"Num answers not equal to {max_num_answers}: {len(new_value_to_add[-1])} for sub: {sub} and key: {key}"
                        if max_answers_dict[sub] < len(new_value_to_add[-1]):
                            max_answers_dict[sub] = len(new_value_to_add[-1])

                    value_to_add = new_value_to_add
                data['data'][sub][key].append(value_to_add)

        # data['data'][sub]['correctness'] = np.array(data['data'][sub]['correctness']).T.astype(float)


        for key in keys_to_add:
            if key == 'predictions':
                data['data'][sub][key] = np.array(data['data'][sub][key]).transpose(1, 0, 2).astype(float)
            else:
                data['data'][sub][key] = np.array(data['data'][sub][key]).T.astype(float)

    return data, max_answers_dict

    # print(max(max_answers_dict.values()))
