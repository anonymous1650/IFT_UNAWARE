import datetime
from ctypes import c_ulong
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ALL_CONFS = []
ALL_IFRS = []
ALL_CIS95 = []
ALL_TEST_NUMS = []
ALL_FLAG = []
NUM = 0

def detect_convergence(data, tolerance=0.001, k=5):
    cumulative = np.cumsum(data) / np.arange(1, len(data) + 1)
    for i in range(k, len(cumulative)):
        window = cumulative[i - k + 1:i + 1]
        if np.max(window) - np.min(window) < tolerance:
            return i + 1, cumulative[i]
    return None, None

def compute_ind_fairness_rate_growth(df, conf, num_samples):

    total_tests, disc_count = 0, 0
    ifr_growth = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= num_samples:
            break

        conf0, conf1 = row["proxy_conf_0"], row["proxy_conf_1"]
        pred_prob0, pred_prob1 = row["pred_prob_0"], row["pred_prob_1"]

        if (conf0 > conf and conf1 < (1 - conf)) or (conf1 > conf and conf0 < (1 - conf)):
            total_tests += 1

            if pred_prob0 != pred_prob1:
                disc_count += 1

            if_rate = disc_count / total_tests
            ifr_growth.append(if_rate)

    return ifr_growth

def plot_IFr_growth_by_conf(conf_list, df, num_samples, isSavePlot=True):

    cumulatives = list(map(lambda conf: compute_ind_fairness_rate_growth(df, conf, num_samples), conf_list))

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(conf_list)))  

    max_num_test = 30  

    for (conf, cumulative), color in zip(zip(conf_list, cumulatives), colors):
        cumulative = np.array(cumulative)
        num_test_by_q = len(cumulative)

        if cumulative.size == 0 or num_test_by_q <= 30: 
            continue

        cumulative_plot = cumulative[29:]  
        x_values = np.arange(30, 30 + len(cumulative_plot))
        start_IFr = cumulative_plot[0] * 100
        last_IFr = cumulative_plot[-1] * 100

        plt.plot(
            x_values, cumulative_plot * 100,
            label=f"conf={conf:.2f}, "
                  f"start={start_IFr:.2f}%, "
                  f"final={last_IFr:.2f}%, n={num_test_by_q}",
            color=color
        )

        plt.scatter(x_values[0], start_IFr, color=color, s=60, zorder=5)

        plt.text(
            x_values[0], start_IFr, f"{start_IFr:.2f}%",
            color=color, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right'
        )

        plt.axvline(x=30, color='gray', linestyle='--', linewidth=0.8, zorder=0)

        max_num_test = max(max_num_test, x_values[-1])

    plt.xlim(30, max_num_test + 1)

    xticks = np.arange(30, max_num_test + 1, max(1, (max_num_test - 30) // 10))
    plt.xticks(xticks)

    global EPSILON

    plt.legend(title=FILE_NAME_BODY + "_" + EPSILON + "_" + str(num_samples), fontsize=10, loc='best')

    plt.xlabel("Number of tests")
    plt.ylabel("Estimated IFr (%)")
    plt.title(f"IFr growth over time with different proxy confs (epsilon={EPSILON}, num_samples={num_samples}): " + FILE_NAME_BODY)
    plt.grid(True)
    plt.tight_layout()

    PLOT_DIR = f"plots/plot-IFr-growth-{EPSILON}-{num_samples}/"
    PLOT_FILE_NAME = f"ifr_growth_{DATASET_NAME}_{PROT_ATTR}_{PROXY_MODEL}_{MAIN_MODEL}_{EPSILON}_{num_samples}.png"
    PLOT_FILE_PATH = PLOT_DIR + PLOT_FILE_NAME

    if isSavePlot:
        plt.savefig(PLOT_FILE_PATH)
        print(f"Plot saved in \"{PLOT_FILE_PATH}\"")
    else:
        plt.show()

def calc_conf_interval(successes, n, confidence=0.95):

    p_hat = successes / n

    z_value = stats.norm.ppf((1 + confidence) / 2)

    me = z_value * np.sqrt((p_hat * (1 - p_hat)) / n)

    lower_limit = p_hat - me
    upper_limit = p_hat + me

    return (me, lower_limit, upper_limit)


def calc_conf_interval_df(df, conf, num_samples):
    cumulative = compute_ind_fairness_rate_growth(df, conf, num_samples)
    if not cumulative:
        return (conf, 0, 0, 0, 0, 0)
    num_test, if_rate = len(cumulative), cumulative[-1]
    num_disc = num_test * if_rate
    conf_interval = calc_conf_interval(num_disc, num_test)
    num_non_disc = num_test * (1-if_rate) # n*(1-p)

    return (conf, if_rate, conf_interval[0], num_test, num_disc, num_non_disc)

# conf = 0.6
# calc_conf_interval_df(df, conf)

def print_IFr_conf_intervals_by_confs(df, conf_list, num_samples):
    ifr_confidence_intervals = list(map(lambda conf: calc_conf_interval_df(df, conf, num_samples), conf_list))

    confs = [item[0] for item in ifr_confidence_intervals]
    ifrs = [item[1] for item in ifr_confidence_intervals]
    cis95 = [item[2] for item in ifr_confidence_intervals]
    test_nums = [item[3] for item in ifr_confidence_intervals]
    nums_disc = [item[4] for item in ifr_confidence_intervals] 
    nums_non_disc = [item[5] for item in ifr_confidence_intervals]

    ALL_CONFS.append(confs)
    ALL_IFRS.append(ifrs)
    ALL_CIS95.append(cis95)
    ALL_TEST_NUMS.append(test_nums)
    ALL_FLAG.append([]) 

    def get_prefix(ifr_value, cis_value, test_num):
        flag_val = 4 # Default to invalid/excluded category
        prefix = 'xxd' # Default prefix

        if test_num >= 30: # Only assign valid flags if enough tests were run
            if ifr_value == 0 or (0 < cis_value < 0.03):
                 flag_val = 1
                 prefix = 'xxa'
            elif 0.03 <= cis_value < 0.06:
                 flag_val = 2
                 prefix = 'xxb'
            elif 0.06 <= cis_value < 0.09:
                 flag_val = 3
                 prefix = 'xxc'
            # else: CI > 0.09 remains 'xxd' and flag 4

        ALL_FLAG[NUM].append(flag_val) # Store the determined flag
        return prefix

    prefixes = [get_prefix(ifr, c, t) for ifr, c, t in zip(ifrs, cis95, test_nums)]

    worst_ifr = None
    
    for i in range(len(confs) - 1, -1, -1):
        if prefixes[i] != 'xxd': 
            worst_ifr = ifrs[i]
            break 

    diff_from_worst = []
    valid_diffs_abs = [] 
    for i in range(len(ifrs)):
        if worst_ifr is not None and prefixes[i] != 'xxd': 
            diff = ifrs[i] - worst_ifr
            diff_from_worst.append(diff)
            valid_diffs_abs.append(abs(diff))
        else:
            diff_from_worst.append(np.nan) 

    max_abs_diff = max(valid_diffs_abs) if valid_diffs_abs else None

    confs_formatted = [f"{c:.2f}" for c in confs]

    ifrs_formatted = [
        "NA" if t == 0 else f"{i*100:.2f}"
        for i, t in zip(ifrs, test_nums)
    ]

    cis95_formatted = [
        f"{c*100:.2f}" for c in cis95
    ]

    test_nums_formatted = [
        f"{t}" for t in test_nums
    ]

    x_worst_ifr_formatted = []
    for i in range(len(diff_from_worst)):
        d = diff_from_worst[i]
        if pd.isna(d): 
            x_worst_ifr_formatted.append("NA")
        else:
            x_worst_ifr_formatted.append(f'{d*100:.2f}')

    # print("|\tConf-level:,\t" + ", ".join(confs_formatted)) 
    print("|\tIFr:         \t" + ", ".join(ifrs_formatted))
    print("|\tConf-Int-95%:\t" + ", ".join(cis95_formatted))
    print("|\tTest-num:    \t" + ", ".join(test_nums_formatted))
    print("|\tdelta:  \t" + ", ".join(x_worst_ifr_formatted)) 

# display_conf_interval_conf_list(df, conf_list)

EPSILON = "e1"  # "e1" or "e2" or "e3" or "e4"
NUM_SAMPLES = 50000
INPUT_FILE_DIR = f"exp_individual_results_{EPSILON}/exp_"
FILE_POSTFIX = ".csv"
DATASET_NAME = "adult" # NO NEED TO EDIT
PROT_ATTR = "gender" # NO NEED TO EDIT
PROXY_MODEL="dnn"
MAIN_MODEL="dnn"
FILE_NAME_BODY = DATASET_NAME+ "_" + PROT_ATTR + "_" + PROXY_MODEL + "_" + MAIN_MODEL
INPUT_FILE_PATH = INPUT_FILE_DIR + FILE_NAME_BODY + FILE_POSTFIX
# print("INPUT_FILE_PATH: ", INPUT_FILE_PATH)

CONF_LIST = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9]

# CSV with header; ie, header=0
df = pd.read_csv(INPUT_FILE_PATH, header=0)
# df.columns = df.columns.str.strip()
# print(df.columns)


from datetime import datetime


def jikken_run_all_configs():
    now = datetime.now()
    print("Log time: ", now.date(), now.time())

    DATASET_NAMEs = ["adult", "bank", "german"]
    PROT_ATTRs = {"adult": ["gender", "race", "age"], "bank":["age"], "german": ["gender", "age"]}
    PROXY_MODELs= ["dnn"]
    MAIN_MODELs= ["dnn"]

    global DATASET_NAME, PROT_ATTR, PROXY_MODEL, MAIN_MODEL, FILE_NAME_BODY, INPUT_FILE_PATH
    global ALL_CONFS, ALL_IFRS, ALL_CIS95, ALL_TEST_NUMS, ALL_FLAG, NUM

    plot_IFr_growth = False
    display_conf_interval = True
    assert(plot_IFr_growth or display_conf_interval)

    for ds_name in DATASET_NAMEs:
        for prot_attr in PROT_ATTRs[ds_name]:
            ALL_CONFS, ALL_IFRS, ALL_CIS95, ALL_TEST_NUMS, ALL_FLAG = [], [], [], [], []
            NUM = 0

            print(ds_name + " " + prot_attr + " " + EPSILON + " " + str(NUM_SAMPLES))
            
            for proxy_model in PROXY_MODELs:
                for main_model in MAIN_MODELs:
                    DATASET_NAME, PROT_ATTR, PROXY_MODEL, MAIN_MODEL \
                        = ds_name, prot_attr, proxy_model, main_model
                    FILE_NAME_BODY = DATASET_NAME + "_" + PROT_ATTR + "_" + PROXY_MODEL + "_" + MAIN_MODEL
                    INPUT_FILE_PATH = INPUT_FILE_DIR + FILE_NAME_BODY + FILE_POSTFIX
                    
                    try:
                        df = pd.read_csv(INPUT_FILE_PATH, header=0)
                    except FileNotFoundError:
                        print(f"[Warning] File not found, skipping: {INPUT_FILE_PATH}")
                        ALL_CONFS.append([0]*len(CONF_LIST))
                        ALL_IFRS.append([0]*len(CONF_LIST))
                        ALL_CIS95.append([0]*len(CONF_LIST))
                        ALL_TEST_NUMS.append([0]*len(CONF_LIST))
                        ALL_FLAG.append([4]*len(CONF_LIST))
                        NUM = NUM + 1
                        continue
                    
                    if(plot_IFr_growth):
                        plot_IFr_growth_by_conf(CONF_LIST, df, NUM_SAMPLES,True)
                    elif(display_conf_interval):
                        print_IFr_conf_intervals_by_confs(df, CONF_LIST, NUM_SAMPLES)
                    
                    NUM = NUM + 1
            
            IFr_ave = [[],[],[]] 
            count = [[],[],[]]  
            cumulative_count = [[],[],[]] 

            for conf in range(len(CONF_LIST)): # 0-8
   
                non_cumulative_counts = [0, 0, 0] # [<3%, >=3% & <6%, >=6% & <9%]
                sums_by_bracket = [0.0, 0.0, 0.0]

                for testcase in range(NUM): # 0-15
                    flag = ALL_FLAG[testcase][conf]
                    ifr = ALL_IFRS[testcase][conf]

                    if flag == 1: # CI < 3%
                        non_cumulative_counts[0] += 1
                        sums_by_bracket[0] += ifr
                    elif flag == 2: # 3% <= CI < 6%
                        non_cumulative_counts[1] += 1
                        sums_by_bracket[1] += ifr
                    elif flag == 3: # 6% <= CI < 9%
                        non_cumulative_counts[2] += 1
                        sums_by_bracket[2] += ifr

                cumulative_counts_conf = [
                    non_cumulative_counts[0],                                  # < 3%
                    non_cumulative_counts[0] + non_cumulative_counts[1],       # < 6%
                    non_cumulative_counts[0] + non_cumulative_counts[1] + non_cumulative_counts[2] # < 9%
                ]
                cumulative_sums_conf = [
                    sums_by_bracket[0],                                        # < 3%
                    sums_by_bracket[0] + sums_by_bracket[1],                   # < 6%
                    sums_by_bracket[0] + sums_by_bracket[1] + sums_by_bracket[2] # < 9%
                ]

                for i in range(3): # 0: <3%, 1: <6%, 2: <9%
                    if cumulative_counts_conf[i] > 0:
                        avg = cumulative_sums_conf[i] / cumulative_counts_conf[i]
                        IFr_ave[i].append(f"{avg*100:.2f}")
                    else:
                        IFr_ave[i].append("NA")

                    cumulative_count[i].append(str(cumulative_counts_conf[i]))

            print("CI95 < 3%:    ", end="")
            print(', '.join(cumulative_count[0])) 
            print("CI95 < 6%:    ", end="")
            print(', '.join(cumulative_count[1])) 
            print("CI95 < 9%:    ", end="")
            print(', '.join(cumulative_count[2]))
            print("")


# ifr_seq = compute_ind_fairness_rate_growth(df,0.5)
# print(ifr_seq)
# point, conv_val = detect_convergence(ifr_seq,tolerance=0.0001, k=20)
# print(point, conv_val)

jikken_run_all_configs()

def calculate_summary_stats():

    print()
    print("Run,Mean,SD,CV")

    means, stds, cvs = [], [], []

    for i in range(len(ALL_IFRS)):
        ifr_list = ALL_IFRS[i]
        flag_list = ALL_FLAG[i]

        filtered_ifrs = [
            ifr for ifr, flag in zip(ifr_list, flag_list) if flag in [1, 2, 3]
        ]

        if filtered_ifrs:
            mean_ifr = np.mean(filtered_ifrs)
            
            if len(filtered_ifrs) == 1:
                std_ifr = 0.0
            else:
                std_ifr = np.std(filtered_ifrs, ddof=1)
            
            if mean_ifr > 0:
                cv = (std_ifr / mean_ifr)
            else:
                cv = np.nan
        else:
            mean_ifr, std_ifr, cv = np.nan, np.nan, np.nan

        means.append(mean_ifr)
        stds.append(std_ifr)
        cvs.append(cv)
        
        run_num = str(i + 1)
        mean_str = f"{mean_ifr*100:.2f}" if pd.notna(mean_ifr) else "NA"
        std_str = f"{std_ifr*100:.2f}" if pd.notna(std_ifr) else "NA"
        cv_str = f"{cv:.2f}" if pd.notna(cv) else "NA"
        
        row_data = [run_num, mean_str, std_str, cv_str]
        print(",".join(row_data))
    
    avg_mean = np.nanmean(means)
    avg_std = np.nanmean(stds)
    avg_cv = np.nanmean(cvs)
    
    avg_mean_str = f"{avg_mean*100:.2f}" if pd.notna(avg_mean) else "NA"
    avg_std_str = f"{avg_std*100:.2f}" if pd.notna(avg_std) else "NA"
    avg_cv_str = f"{avg_cv:.2f}" if pd.notna(avg_cv) else "NA"
    
    avg_row_data = ["Avg", avg_mean_str, avg_std_str, avg_cv_str]
    print(",".join(avg_row_data))

#calculate_summary_stats()
exit()
