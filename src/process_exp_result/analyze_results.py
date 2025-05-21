# run this file at /src: python python_test/my_test.py
import sys
import os
# add src to sys.path, so that we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # os.path.dirname(__file__) is the directory of this file, i.e. /src/process_exp_result
from module import load

def extract_metrics(result_path):
    """
    # Example usage:
    # this result_path is relative to the current working directory, which is /src
    # this means . represents /src
    result_path = os.path.join('.', 'output', 'result', '0_GUE_EMP_H3_dnabert2_0_none_EMP_H3')  
    """
    result = load(result_path)
    metrics = {
        'tag': result['cfg']['tag'],
        "task_name": result['cfg']['control']['task_name'],
        "subset_name": result['cfg']['control']['subset_name'],
        "valid_acc": result['logger']['test']['mean']['valid/Accuracy'],
        "valid_F1": result['logger']['test']['mean']['valid/F1'],
        "valid_MCC": result['logger']['test']['mean']['valid/MCC'],
        "test_acc": result['logger']['test']['mean']['test/Accuracy'],
        "test_F1": result['logger']['test']['mean']['test/F1'],
        "test_MCC": result['logger']['test']['mean']['test/MCC'],
    }
    return metrics

def get_exp_results(result_dir):
    """
    Get all experiment results from the result directory.
    :param result_dir: The directory containing the experiment results.
    :return: A list of tuples containing the metrics for each experiment.
    """
    exp_results = []
    for filename in os.listdir(result_dir):
        result_path = os.path.join(result_dir, filename)
        metrics = extract_metrics(result_path)
        exp_results.append(metrics)
    return exp_results



# path is relative to the current working directory, which is /src
# this means . means /src
result_dir = os.path.join('.', 'output', 'result')  # relative to src/process_exp_result
exp_results = get_exp_results(result_dir)

print("Done")

