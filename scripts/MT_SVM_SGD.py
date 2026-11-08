
# %% --- Load Modules -----------------------------------------------------------------------------

print("Loading modules")

from itertools import zip_longest
from numpy import ceil, round, concatenate, load as np_load
from numpy.random import default_rng
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from time import perf_counter
from datetime import datetime
from pandas import DataFrame
from pickle import dump
import numpy as np

from _MT_functions import get_arguments, get_classes_training, get_settings_summary, get_subjects, get_split_files

flo_time_total_start = perf_counter()

# %% --- Handle Arguments -------------------------------------------------------------------------

print("Getting the arguments")

args = get_arguments()

# %% --- Split Subjects Into Training- and Test-Set -----------------------------------------------

lis_subjects = get_subjects(pat_data=args.pat_data)
int_n_subjects_total = len(lis_subjects)

lis_files_train, lis_files_test, lis_subjects_train, lis_subjects_test = get_split_files()
int_subjects_train_percent = int(100*(len(lis_subjects_train)/400)) # 400 -> max. number of training subjects

pat_results_svm = args.pat_data.parent/"results"/"SVM"

df_results = DataFrame(columns=["Epoch", "Accuracy_Train", "Accuracy_Test", "Balanced_Accuracy_Train", "Balanced_Accuracy_Test", "Precision_Train", "Precision_Test", "Recall_Train", "Recall_Test", "F1-Score_Train", "F1-Score_Test"])
df_subjects = DataFrame(zip_longest(lis_subjects_train, lis_subjects_test), columns=["Train", "Test"])

rng = default_rng(seed=args.int_seed)

# %% --- Setup the Dataloader ---------------------------------------------------------------------

print("Seting up the dataloader")

class DataLoader:

    def __init__(self, files:list, batch_size:int = 1, shuffle_data:bool = True):

        self.files = files
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data

        self.len_files = len(self.files)
    
    def __iter__(self):

        if self.shuffle_data:

            rng.shuffle(self.files)

        for batch_start in range(0, self.len_files, self.batch_size):

            batch_files = self.files[batch_start:batch_start+self.batch_size]

            lis_batch_data = []
            lis_batch_labels = []

            for pat_file in batch_files:

                with np_load(pat_file, allow_pickle=True) as file:

                    lis_batch_data.append(file["data"])
                    lis_batch_labels.append(file["labels"])

            arr_batch_data = concatenate(lis_batch_data, axis=0)
            arr_batch_labels = concatenate(lis_batch_labels, axis=0)

            yield arr_batch_data, arr_batch_labels

dl_train = DataLoader(files=lis_files_train, batch_size=args.int_batch_size_train, shuffle_data=True)
dl_test = DataLoader(files=lis_files_test, batch_size=args.int_batch_size_test, shuffle_data=False)

str_classifier_loss = "hinge"
str_classifier_penalty = "l2"
str_classifier_learning_rate = "optimal"
sgd_partial = SGDClassifier(loss=str_classifier_loss, penalty=str_classifier_penalty, learning_rate=str_classifier_learning_rate, random_state=args.int_seed, n_jobs=args.int_n_jobs)
 
# %% 

lis_classes_train, lis_classes_coded_train = get_classes_training()

get_settings_summary(args=args, int_n_subjects_total=int_n_subjects_total, lis_classes_train = lis_classes_train, pat_results = pat_results_svm)

print("--- Begin Training -------------------------------------------------------------------------")
print("")

int_n_train_batches = int(ceil(len(dl_train.files) / dl_train.batch_size))
int_n_test_batches = int(ceil(len(dl_test.files) / dl_test.batch_size))

dt_now = datetime.now()

for int_idx_epoch, int_epoch in enumerate(range(args.int_epochs)):

    flo_time_epoch_start = perf_counter()

    print(f"\33[1mEpoch: {int_epoch + 1}/{args.int_epochs}\33[0m")

    int_n_train_batch = 1
    int_n_test_batch = 1

    # --- Training ------------------------------

    flo_time_training_start = perf_counter()

    lis_train_trues = []
    lis_train_preds = []

    for train_data_batch, train_labels_batch in dl_train:

        if int_epoch == 0 and int_n_train_batch == 1:      

            sgd_partial.partial_fit(X=train_data_batch, y=train_labels_batch, classes= lis_classes_coded_train)

        else:

            sgd_partial.partial_fit(X=train_data_batch, y=train_labels_batch)

        train_batch_pred = sgd_partial.predict(X=train_data_batch)

        lis_train_trues.append(train_labels_batch)
        lis_train_preds.append(train_batch_pred)

        # Progress Bar
        int_batch_batches_ratio = int(round((int_n_train_batch * args.int_progress_bar_size)/int_n_train_batches, decimals=1)*10)
        str_train_progress_bar = "[" + "▌" * int_batch_batches_ratio  + "." * ((args.int_progress_bar_size*10) - int_batch_batches_ratio) + "]"
        print(f"  Training Batches:  {f"{int_n_train_batch}/{int_n_train_batches}":>3} " + str_train_progress_bar, end="\r")

        int_n_train_batch += 1

    arr_train_trues = concatenate(lis_train_trues)
    arr_train_preds = concatenate(lis_train_preds)

    flo_epoch_train_accuracy = accuracy_score(y_true=arr_train_trues, y_pred=arr_train_preds)
    flo_epoch_train_balanced_accuracy = balanced_accuracy_score(y_true=arr_train_trues, y_pred=arr_train_preds)
    flo_epoch_train_precision = precision_score(y_true=arr_train_trues, y_pred=arr_train_preds, average="macro", zero_division=np.nan)
    flo_epoch_train_recall = recall_score(y_true=arr_train_trues, y_pred=arr_train_preds, average="macro", zero_division=np.nan)
    flo_epoch_train_f1_score = f1_score(y_true=arr_train_trues, y_pred=arr_train_preds, average="macro", zero_division=np.nan)
    
    flo_time_training_end = perf_counter()
 
    print("")
    
    # --- Testing -------------------------------

    flo_time_testing_start = perf_counter()

    lis_test_trues = []
    lis_test_preds = []

    for test_data_batch, test_labels_batch in dl_test:

        test_batch_pred = sgd_partial.predict(X=test_data_batch)

        lis_test_trues.append(test_labels_batch)
        lis_test_preds.append(test_batch_pred)

        # Progress Bar
        int_batch_batches_ratio = int(round((int_n_test_batch * args.int_progress_bar_size) / int_n_test_batches, decimals=1)*10)
        str_test_progress_bar = "[" + "▌" * int_batch_batches_ratio + "." * ((args.int_progress_bar_size*10) - int_batch_batches_ratio) + "]"
        print(f"  Testing Batches:  {f"{int_n_test_batch}/{int_n_test_batches}":>4} " + str_test_progress_bar, end="\r")   

        int_n_test_batch += 1

    arr_test_trues = concatenate(lis_test_trues)
    arr_test_preds = concatenate(lis_test_preds)

    flo_epoch_test_accuracy = accuracy_score(y_true=arr_test_trues, y_pred=arr_test_preds)
    flo_epoch_test_balanced_accuracy = balanced_accuracy_score(y_true=arr_test_trues, y_pred=arr_test_preds)
    flo_epoch_test_precision = precision_score(y_true=arr_test_trues, y_pred=arr_test_preds, average="macro", zero_division=np.nan)
    flo_epoch_test_recall = recall_score(y_true=arr_test_trues, y_pred=arr_test_preds, average="macro", zero_division=np.nan)
    flo_epoch_test_f1_score = f1_score(y_true=arr_test_trues, y_pred=arr_test_preds, average="macro", zero_division=np.nan)

    df_results.loc[int_idx_epoch, "Epoch"] = int_epoch + 1
    df_results.loc[int_idx_epoch, "Accuracy_Train"] = flo_epoch_train_accuracy
    df_results.loc[int_idx_epoch, "Accuracy_Test"] = flo_epoch_test_accuracy
    df_results.loc[int_idx_epoch, "Balanced_Accuracy_Train"] = flo_epoch_train_balanced_accuracy
    df_results.loc[int_idx_epoch, "Balanced_Accuracy_Test"] = flo_epoch_test_balanced_accuracy
    df_results.loc[int_idx_epoch, "Precision_Train"] = flo_epoch_train_precision
    df_results.loc[int_idx_epoch, "Precision_Test"] = flo_epoch_test_precision
    df_results.loc[int_idx_epoch, "Recall_Train"] = flo_epoch_train_recall
    df_results.loc[int_idx_epoch, "Recall_Test"] = flo_epoch_test_recall
    df_results.loc[int_idx_epoch, "F1-Score_Train"] = flo_epoch_train_f1_score
    df_results.loc[int_idx_epoch, "F1-Score_Test"] = flo_epoch_test_f1_score

    # ---------------------------------------------------------------------------------------------

    flo_time_testing_end = perf_counter()
    flo_time_epoch_end = perf_counter()
    flo_time_total_end = perf_counter()

    flo_time_total = flo_time_total_end - flo_time_total_start
    flo_time_epoch = flo_time_epoch_end - flo_time_epoch_start
    flo_time_training = flo_time_training_end - flo_time_training_start
    flo_time_testing = flo_time_testing_end - flo_time_testing_start

    # ---------------------------------------------------------------------------------------------

    df_predictions = DataFrame(zip_longest(arr_test_preds, arr_test_trues), columns=["Predicted State", "True State"])

    str_folder_name = f"{dt_now.day}-{dt_now.month}-{dt_now.year}_{dt_now.hour}-{dt_now.minute}-{dt_now.second}_{args.lis_tasks_test[0]}_{int_subjects_train_percent}"
    pat_results_svm_folder = pat_results_svm/str_folder_name

    if not pat_results_svm_folder.is_dir():

        pat_results_svm_folder.mkdir(parents=True, exist_ok=True)

    str_file_name_results = f"{str_folder_name}_performance.csv"
    str_file_name_subjects = f"{str_folder_name}_subjects.csv"
    str_file_name_predictions = f"{str_folder_name}_epoch_{int_epoch + 1}_predictions.csv"
    str_file_name_model = f"{str_folder_name}_epoch_{int_epoch + 1}_model.pkl"
    str_file_name_parameters = f"{str_folder_name}_parameters.csv"

    df_results.to_csv(path_or_buf=pat_results_svm_folder/str_file_name_results, index=False)
    df_subjects.to_csv(path_or_buf=pat_results_svm_folder/str_file_name_subjects, index=False)
    df_predictions.to_csv(path_or_buf=pat_results_svm_folder/str_file_name_predictions, index=False)

    with open(pat_results_svm_folder/str_file_name_model, "wb") as file_model:
        dump(sgd_partial, file_model)

    print("")
    print("")
    print(f"  \33[1mTraining:\33[0m")
    print(f"    Accuracy: {f"{100*flo_epoch_train_accuracy:.2f}":>9}%")
    print(f"    Precision: {f"{100*flo_epoch_train_precision:.2f}":>8}%")
    print(f"    Recall: {f"{100*flo_epoch_train_recall:.2f}":>11}%")
    print(f"    F1-Score: {f"{100*flo_epoch_train_f1_score:.2f}":>9}%")
    print("")
    print(f"  \33[1mTesting:\33[0m")
    print(f"    Accuracy: {f"{100*flo_epoch_test_accuracy:.2f}":>9}%")
    print(f"    Precision: {f"{100*flo_epoch_test_precision:.2f}":>8}%")
    print(f"    Recall: {f"{100*flo_epoch_test_recall:.2f}":>11}%")
    print(f"    F1-Score: {f"{100*flo_epoch_test_f1_score:.2f}":>9}%")
    print("")
    print(f"  \33[1mTime:\33[0m")
    print(f"    Epoch: {f"{flo_time_epoch:.3f}":>12}s")
    print(f"    Training: {f"{flo_time_training:.3f}":>9}s")
    print(f"    Testing: {f"{flo_time_testing:.3f}":>10}s")

    print("")

df_parameters = DataFrame(columns=["Parameter"], index=["Epochs", "Subjects_training", "Subjects_testing", "Batch_size_training", "Batch_size_testing", "Tasks_training", "Tasks_testing", "Categories", "Baseline", "SGD_loss", "SGD_penalty", "SGD_learning_rate", "Random_seed", "n_jobs"])

df_parameters.loc["Epochs", "Parameter"] = args.int_epochs
df_parameters.loc["Subjects_training", "Parameter"] = args.int_n_subjects_train
df_parameters.loc["Subjects_testing", "Parameter"] = args.int_n_subjects_test
df_parameters.loc["Batch_size_training", "Parameter"] = args.int_batch_size_train
df_parameters.loc["Batch_size_testing", "Parameter"] = args.int_batch_size_test
df_parameters.loc["Tasks_training", "Parameter"] = args.lis_tasks_train
df_parameters.loc["Tasks_testing", "Parameter"] = args.lis_tasks_test
df_parameters.loc["Categories", "Parameter"] = len(lis_classes_train)
df_parameters.loc["Baseline", "Parameter"] = 100*(1/len(lis_classes_train))
df_parameters.loc["SGD_loss", "Parameter"] = str_classifier_loss
df_parameters.loc["SGD_penalty", "Parameter"] = str_classifier_penalty
df_parameters.loc["SGD_learning_rate", "Parameter"] = str_classifier_learning_rate
df_parameters.loc["Random_seed", "Parameter"] = args.int_seed
df_parameters.loc["n_jobs", "Parameter"] = args.int_n_jobs

df_parameters.to_csv(path_or_buf=pat_results_svm_folder/str_file_name_parameters, index=True)

print(f"    Total: {f"{flo_time_total:.3f}":>10}s")
print("")

# %%




# --- Plotting ------------------------------------------------------------------------------------

# subjects training
# subjects test
# streuung bei geringerer datensatzgrüße
# paper john, haufe - svm interpretation weights

#TODO in paper: discuss fit vs partial fit, performance, memory, time and partial fit as a way to analyze large fmri datasets
#TODO reflect on temporal autocorrelation in the paper, what is it?, how does it relate to fMRI? what does it mean for classifiers (SVM and deep learning), prolem if you "rip" subject data appart

#TODO meantion, explain and discuss that batches in training are less well trained and the start and better trained at the end -> mean could be misleading
#TODO explain metrics + balanced accuracy vs accuracy