# run this file at /src: python python_test/my_test.py
import sys
import os
import pandas as pd
import re
import argparse

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
        "subset_name_test":result['cfg']['control']['subset_name_test'],
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
    pattern = r'^\d+_.+' # <number>_<...> file name pattern for exp results
    for filename in os.listdir(result_dir):
        if re.match(pattern, filename):
            result_path = os.path.join(result_dir, filename)
            metrics = extract_metrics(result_path)
            exp_results.append(metrics)
    return exp_results


if __name__ == '__main__':
    # Example usage: python process_exp_result/analyze_results.py --result_dir ./output/result_test

    parser = argparse.ArgumentParser(description="Extract experiment results and export to Excel.")
    parser.add_argument(
        "--result_dir",
        type=str,
        # path is relative to the current working directory, which is /src
        # this means . means /src
        default=os.path.join('.', 'output', 'result'),  # default value
        help="Path to the result directory (suppose you are at the current working directory, i.e., /src)"
    )
    args = parser.parse_args()
    
    result_dir = args.result_dir
    exp_results = get_exp_results(result_dir)

    # print(exp_results)

    # save as excel
    excel_path = os.path.join(result_dir, 'exp_results.xlsx')
    df = pd.DataFrame(exp_results)
    df.to_excel(excel_path, index=False)

    print(f"Results saved to: {excel_path}")

