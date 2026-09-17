
# %% --- Load Modules -----------------------------------------------------------------------------

import argparse
from numpy import array, ceil, round, concatenate, load, mean, append
from numpy.random import shuffle
from os import listdir
from os.path import join
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
from datetime import datetime, timezone

from _MT_functions import compute_bold_files, get_subjects

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = argparse.ArgumentParser()

ap.add_argument("--files", required=True)
ap.add_argument("--force_recompute_files", default=False, required=False)
ap.add_argument("--verbose", default=False, required=False)

#args = ap.parse_args()

#str_files_path = str(args.files)
#boo_force_recompute_files = bool(args.force_recompute_files)
#boo_verbose = bool(args.verbose)

# %%

# parameters for function
str_data_path = "/home/daniel/Desktop/Python/Data/MRI/HCP_AWS"
str_data_type = "bold"
flo_train_size = 0.8
verbose = False
int_progress_bar_size = 4
int_random_state = 42
flo_signal = 0.1


# %%

compute_bold_files(data_path=str_data_path, boo_standardize=False, verbose=False, force_new_files=False)

lis_subjects = get_subjects(data_path=str_data_path)

lis_train_subjects, lis_test_subjects = train_test_split(lis_subjects, train_size=flo_train_size, random_state=int_random_state)

lis_train_files = []
lis_test_files = []

for subject in lis_subjects:

    if str_data_type == "bold":

        str_subject_path = join(str_data_path, subject, "func/")
        lis_subject_files = [x for x in listdir(path=str_subject_path) if x.endswith("bold.npz")]

    if str_data_type == "betas":

        str_subject_path = join(str_data_path, subject, "func", "betas")
        lis_subject_files = [x for x in listdir(str_subject_path) if x.endswith("betas.npz")]

    for file in lis_subject_files:

        if subject in lis_train_subjects:

            lis_train_files.append(file)

        if subject in lis_test_subjects:

            lis_test_files.append(file)

class DataLoader:

    def __init__(self, files:list, batch_size:int = 4, shuffle_data:bool = True, boo_standardize:bool=True):

        self.files = files
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.standardize = boo_standardize

        if self.shuffle_data:

            shuffle(self.files)

        self.len_files = len(self.files)
    
    def __iter__(self):

        for batch_start in range(0, self.len_files, self.batch_size):

            batch_files = self.files[batch_start:batch_start+self.batch_size]

            lis_batch_data = []
            lis_batch_labels = []

            for file in batch_files:

                str_subject = file.split("_")[0]

                if str_data_type == "bold":

                    str_file_path = join(str_data_path, f"{str_subject}", "func", file)

                if str_data_type == "betas":

                    str_file_path = join(str_data_path, f"{str_subject}", "func", "betas", file)

                with load(str_file_path) as file:

                    if self.standardize:

                        scaler = StandardScaler()
                        scaler.fit_transform(file["data"])

                    lis_batch_data.append(file["data"])

                    # TODO works but make the way it deals with beta labels and bold labels less ugly, maybe change the way labels are saved
                    if str_data_type == "bold":

                        lis_batch_labels.append(file["labels"])

                    if str_data_type == "betas":

                        lis_batch_labels.append([file["labels"].item()])

            arr_batch_data = concatenate(lis_batch_data, axis=0)
            arr_batch_labels = concatenate(lis_batch_labels, axis=0)

            yield arr_batch_data, arr_batch_labels

dl_train_dataloader = DataLoader(files=lis_train_files, batch_size=1, shuffle_data=True, boo_standardize=True)
dl_test_dataloader = DataLoader(files=lis_test_files, batch_size=1, shuffle_data=True, boo_standardize=True)

# remember: batch_size = 1 for bold means shape(143,xxx) and for betas means shape(1,xxx)
 
# %%

int_epochs = 3

sgd_partial = SGDClassifier(loss="hinge", penalty="l2", learning_rate="optimal" , random_state=42)

lis_classes = [0,1]  #TODO improve

arr_epochs_train_accuracies = []
arr_epochs_test_accuracies = []

int_n_train_batches = int(ceil(len(dl_train_dataloader.files) / dl_train_dataloader.batch_size))
int_n_test_batches = int(ceil(len(dl_test_dataloader.files) / dl_test_dataloader.batch_size))

# --- Training ----------------------------------

for int_epoch in range(int_epochs):

    str_date_time = datetime.now(tz=timezone.utc).isoformat().replace("T", " ")

    flo_time_epoch_start = time()
    flo_time_training_start = time()

    print(f"Epoch: {int_epoch + 1}/{int_epochs}")

    int_n_train_batch = 1
    int_n_test_batch = 1

    int_format_width = 9

    arr_batches_train_accuracies = array([])
    arr_batches_test_accuracies = array([])

    for train_data_batch, train_labels_batch in dl_train_dataloader:

        if int_epoch == 0:      

            sgd_partial.partial_fit(X=train_data_batch, y=train_labels_batch, classes= lis_classes)

        else:

            sgd_partial.partial_fit(X=train_data_batch, y=train_labels_batch)

        batch_pred = sgd_partial.predict(X=train_data_batch)

        flo_train_batch_accuracy = accuracy_score(y_true=train_labels_batch, y_pred=batch_pred)

        arr_batches_train_accuracies = append(arr=arr_batches_train_accuracies, values=flo_train_batch_accuracy)

        # Progress Bar
        int_batch_batches_ratio = int(round((int_n_train_batch * int_progress_bar_size)/int_n_train_batches, decimals=1)*10)
        str_train_progress_bar = "[" + "▌" * int_batch_batches_ratio  + "." * ((int_progress_bar_size*10) - int_batch_batches_ratio) + "]"
        print(f"  Training Batches:  {f"{int_n_train_batch}/{int_n_train_batches}":>7} " + str_train_progress_bar, end="\r")

        int_n_train_batch += 1

    arr_epochs_train_accuracies = append(arr=arr_epochs_train_accuracies, values=arr_batches_train_accuracies.mean())

    flo_time_training_end = time()

    print("")
    
    # --- Testing -------------------------------

    flo_time_testing_start = time()

    for test_data_batch, test_labels_batch in dl_test_dataloader:

        batch_pred = sgd_partial.predict(X=test_data_batch)

        flo_test_batch_accuracy = accuracy_score(y_true=test_labels_batch, y_pred=batch_pred)

        arr_batches_test_accuracies = append(arr=arr_batches_test_accuracies, values=flo_test_batch_accuracy)

        # Progress Bar
        int_batch_batches_ratio = int(round((int_n_test_batch * int_progress_bar_size) / int_n_test_batches, decimals=1)*10)
        str_test_progress_bar = "[" + "▌" * int_batch_batches_ratio + "." * ((int_progress_bar_size*10) - int_batch_batches_ratio) + "]"
        print(f"  Testing Batches:  {f"{int_n_test_batch}/{int_n_test_batches}":>8} " + str_test_progress_bar, end="\r")   

        int_n_test_batch += 1

    arr_epochs_test_accuracies= append(arr=arr_epochs_test_accuracies, values=arr_batches_test_accuracies.mean())

    flo_time_testing_end = time()
    flo_time_epoch_end = time()

    flo_time_epoch = flo_time_epoch_end - flo_time_epoch_start
    flo_time_training = flo_time_training_end - flo_time_training_start
    flo_time_testing = flo_time_testing_end - flo_time_testing_start

    print("")
    print("")

    print(f"  Training Accuracy:  {(arr_epochs_train_accuracies[-1]*100):.2f}%")
    print(f"  Test Accuracy:  {f"{(arr_epochs_test_accuracies[-1]*100):.2f}":>9}%")
    print("")
    print(f"  Time: {flo_time_epoch:.0f}s")

    print("")


# --- Plotting ------------------------------------------------------------------------------------



# subjects training
# subjects test
# streuung bei geringerer datensatzgrüße
# paper john, haufe - svm interpretation weights

#TODO in paper: discuss fit vs partial fit, performance, memory, time and partial fit as a way to analyze large fmri datasets
#TODO reflect on temporal autocorrelation in the paper, what is it?, how does it relate to fMRI? what does it mean for classifiers (SVM and deep learning), prolem if you "rip" subject data appart


# %%

# detrending ? yes/no?

# svm whole brain vs ROI

