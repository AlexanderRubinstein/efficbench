import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
import os


CACHE_DIR = "./cache_dir"


MODELS_NAMES  = ['logicker/SkkuDataScienceGlobal-10.7b',
        'yujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v2',
        'kyujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v1',
        'fblgit/UNA-POLAR-10.7B-InstructMath-v2',
        'abacusai/MetaMath-bagel-34b-v0.2-c1500',
        'Q-bert/MetaMath-Cybertron-Starling',
        'meta-math/MetaMath-70B-V1.0',
        'meta-math/MetaMath-Mistral-7B',
        'SanjiWatsuki/neural-chat-7b-v3-3-wizardmath-dare-me',
        'ed001/datascience-coder-6.7b',
        'abacusai/Fewshot-Metamath-OrcaVicuna-Mistral',
        'meta-math/MetaMath-70B-V1.0',
        'WizardLM/WizardMath-70B-V1.0',
        'WizardLM/WizardMath-13B-V1.0',
        'WizardLM/WizardMath-7B-V1.0',
        'SanjiWatsuki/neural-chat-7b-v3-3-wizardmath-dare-me',
        'meta-math/MetaMath-Llemma-7B',
        'rameshm/llama-2-13b-mathgpt-v4',
        'rombodawg/Everyone-Coder-4x7b-Base',
        'qblocks/mistral_7b_DolphinCoder',
        'FelixChao/vicuna-33b-coder',
        'rombodawg/LosslessMegaCoder-llama2-13b-mini',
        'defog/sqlcoder-34b-alpha',
        'WizardLM/WizardCoder-Python-34B-V1.0',
        'OpenBuddy/openbuddy-deepseekcoder-33b-v16.1-32k',
        'mrm8488/llama-2-coder-7b',
        'jondurbin/airocoder-34b-2.1',
        'openchat/opencoderplus',
        'bigcode/starcoderplus',
        'qblocks/falcon_7b_DolphinCoder',
        'deepseek-ai/deepseek-coder-6.7b-instruct',
        'ed001/datascience-coder-6.7b',
        'glaiveai/glaive-coder-7b',
        'uukuguy/speechless-coder-ds-6.7b',
        'WizardLM/WizardCoder-Python-7B-V1.0',
        'WizardLM/WizardCoder-15B-V1.0',
        'LoupGarou/WizardCoder-Guanaco-15B-V1.1',
        'GeorgiaTechResearchInstitute/starcoder-gpteacher-code-instruct',
        'deepseek-ai/deepseek-coder-1.3b-instruct',
        'uukuguy/speechless-coder-ds-1.3b',
        'bigcode/tiny_starcoder_py',
        'Deci/DeciCoder-1b',
        'KevinNi/mistral-class-bio-tutor',
        'FelixChao/vicuna-7B-chemical',
        'AdaptLLM/finance-chat',
        'ceadar-ie/FinanceConnect-13B',
        'Harshvir/Llama-2-7B-physics',
        'FelixChao/vicuna-7B-physics',
        'lgaalves/gpt-2-xl_camel-ai-physics',
        'AdaptLLM/law-chat'
]


SCENARIOS = ['harness_arc_challenge_25',
             'harness_hellaswag_10',
             #'harness_hendrycksTest_5',
             'harness_truthfulqa_mc_0',
             "harness_winogrande_5",
             "harness_gsm8k_5"]

MMLU_SUBSCENARIOS = ['harness_hendrycksTest_abstract_algebra_5', 'harness_hendrycksTest_anatomy_5',
                     'harness_hendrycksTest_astronomy_5', 'harness_hendrycksTest_business_ethics_5',
                     'harness_hendrycksTest_clinical_knowledge_5', 'harness_hendrycksTest_college_biology_5',
                     'harness_hendrycksTest_college_chemistry_5', 'harness_hendrycksTest_college_computer_science_5',
                     'harness_hendrycksTest_college_mathematics_5', 'harness_hendrycksTest_college_medicine_5',
                     'harness_hendrycksTest_college_physics_5', 'harness_hendrycksTest_computer_security_5',
                     'harness_hendrycksTest_conceptual_physics_5', 'harness_hendrycksTest_econometrics_5',
                     'harness_hendrycksTest_electrical_engineering_5', 'harness_hendrycksTest_elementary_mathematics_5',
                     'harness_hendrycksTest_formal_logic_5', 'harness_hendrycksTest_global_facts_5',
                     'harness_hendrycksTest_high_school_biology_5', 'harness_hendrycksTest_high_school_chemistry_5',
                     'harness_hendrycksTest_high_school_computer_science_5', 'harness_hendrycksTest_high_school_european_history_5',
                     'harness_hendrycksTest_high_school_geography_5', 'harness_hendrycksTest_high_school_government_and_politics_5',
                     'harness_hendrycksTest_high_school_macroeconomics_5', 'harness_hendrycksTest_high_school_mathematics_5',
                     'harness_hendrycksTest_high_school_microeconomics_5', 'harness_hendrycksTest_high_school_physics_5',
                     'harness_hendrycksTest_high_school_psychology_5', 'harness_hendrycksTest_high_school_statistics_5',
                     'harness_hendrycksTest_high_school_us_history_5', 'harness_hendrycksTest_high_school_world_history_5',
                     'harness_hendrycksTest_human_aging_5', 'harness_hendrycksTest_human_sexuality_5',
                     'harness_hendrycksTest_international_law_5', 'harness_hendrycksTest_jurisprudence_5',
                     'harness_hendrycksTest_logical_fallacies_5', 'harness_hendrycksTest_machine_learning_5',
                     'harness_hendrycksTest_management_5', 'harness_hendrycksTest_marketing_5',
                     'harness_hendrycksTest_medical_genetics_5', 'harness_hendrycksTest_miscellaneous_5',
                     'harness_hendrycksTest_moral_disputes_5', 'harness_hendrycksTest_moral_scenarios_5',
                     'harness_hendrycksTest_nutrition_5', 'harness_hendrycksTest_philosophy_5',
                     'harness_hendrycksTest_prehistory_5', 'harness_hendrycksTest_professional_accounting_5',
                     'harness_hendrycksTest_professional_law_5', 'harness_hendrycksTest_professional_medicine_5',
                     'harness_hendrycksTest_professional_psychology_5', 'harness_hendrycksTest_public_relations_5',
                     'harness_hendrycksTest_security_studies_5', 'harness_hendrycksTest_sociology_5',
                     'harness_hendrycksTest_us_foreign_policy_5', 'harness_hendrycksTest_virology_5',
                     'harness_hendrycksTest_world_religions_5']


LB_SAVEPATH = 'data/leaderboard_fields_raw_20240125.pickle'



def main():

    lb_savepath = LB_SAVEPATH

    models = []
    for m in MODELS_NAMES:
        creator, model = tuple(m.split("/"))
        models.append('open-llm-leaderboard/details_{:}__{:}'.format(creator, model))

    data = {}
    for model in tqdm(models):
        data[model] = {}
        for s in SCENARIOS + MMLU_SUBSCENARIOS:
            data[model][s] = {}
            data[model][s]['correctness'] = None
            data[model][s]['dates'] = None

    os.makedirs(CACHE_DIR, exist_ok=True)
    skipped = 0
    log = []
    for model in tqdm(models):
        skipped_aux=0
        for s in MMLU_SUBSCENARIOS:
            if 'arc' in s: metric = 'acc_norm'
            elif 'hellaswag' in s: metric = 'acc_norm'
            elif 'truthfulqa' in s: metric = 'mc2'
            else: metric = 'acc'

            try:
                aux = load_dataset(model, s, cache_dir=CACHE_DIR)
                data[model][s]['dates'] = list(aux.keys())
                data[model][s]['correctness'] = [a[metric] for a in aux['latest']['metrics']]
                print("\nOK {:} {:}\n".format(model,s))
                log.append("\nOK {:} {:}\n".format(model,s))
            except Exception as e:
                print(f"Error loading dataset for {model} and {s}: {e}")
                try:
                    aux = load_dataset(model, s, cache_dir=CACHE_DIR)
                    data[model][s]['dates'] = list(aux.keys())
                    data[model][s]['correctness'] = aux['latest'][metric]
                    print("\nOK {:} {:}\n".format(model,s))
                    log.append("\nOK {:} {:}\n".format(model,s))
                except Exception as e:
                    print(f"Error loading dataset for {model} and {s}: {e}")
                    data[model][s] = None
                    print("\nSKIP {:} {:}\n".format(model,s))
                    skipped_aux+=1
                    log.append("\nSKIP {:} {:}\n".format(model,s))

        if skipped_aux>0: skipped+=1

        with open(lb_savepath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("\nModels skipped so far: {:}\n".format(skipped))


if __name__ == "__main__":
    main()
