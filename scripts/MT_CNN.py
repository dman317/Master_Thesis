
#from argparse import ArgumentParser

#ap = ArgumentParser()

#ap.add_argument("-data-path", default="")
#ap.add_argument("-epochs", default=50)
#ap.add_argument("-train-test-ratio", default=0.8)
#ap.add_argument("-batch-size", default=12)
#ap.add_argument("-learning_rate", default=0.001)
#ap.add_argument("-dropout", default=0.2)
#ap.add_argument("-random-seed", default=42)

#args = ap.parse_args()

# %%

#def CNN(str_data_type:str, str_data_path:str="/home/daniel/Desktop/Python/Data/MRI/HCP_AWS", int_n_epochs:int = 50, flo_train_test_ratio:float = 0.8, int_batch_size:int=12, flo_learning_rate:float=0.001, flo_dropout:float=0.2, int_random_seed:int=42, boo_verbose:bool=False):

str_data_type = "betas"
str_data_path = "/home/daniel/Desktop/Python/Data/MRI/HCP_AWS"
int_n_epochs = 20
flo_train_test_ratio = 0.8
int_batch_size = 12
flo_learning_rate = 0.001
flo_dropout = 0.2
int_random_seed = 42
boo_verbose = False

# TODO make sure int_batch_size=12 its correct
# TODO investigate test > train issue
# TODO make it one big function with all the parameters in one function definition
# TODO make it work for bold files
# TODO save weights in each epoch in seperate folder
# TODO check grammar of all prints & and decide what is boo_verbose and what not

# --- Load Modules -----------------------------------------------------------------------------

from matplotlib.pyplot import show, subplots
from nibabel import __version__ as nib_version
from nilearn import __version__ as nil_version
from numpy import arange, ceil, mean, round, __version__ as np_version
from os import listdir
from os.path import join
from pandas import __version__ as pd_version
from platform import python_version
from psutil import virtual_memory
from sklearn import __version__ as skl_version
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from time import time
from datetime import datetime
import torch
from torch import __version__ as tor_version
from torch.cuda import is_available
from torch.nn import BCEWithLogitsLoss, BatchNorm3d, Conv3d, Dropout3d, LazyLinear, Linear, Module
from torch.nn.functional import max_pool3d, relu
from torch.utils.data import DataLoader, Dataset

from _MT_functions import get_subjects, compute_betas, handle_logs, get_hardware, progress_bar, remove_log_files, compute_bold_files

device = torch.device("cuda" if is_available() else "cpu")

# ----------------------------------------------------------------------------------------------

lis_subjects = get_subjects(data_path=str_data_path)

dic_log_epochs = {}
dic_log_run = {}
dic_log_subjects = {}

str_log_path = join(str_data_path, "logs")
lis_log_files = [x for x in listdir(str_log_path) if x.endswith("run.csv")]
int_run_id = len(lis_log_files) + 1

str_date = datetime.now().strftime("%d.%m.%Y")
str_time = datetime.now().strftime("%H:%M:%S")

if str_data_type == "betas":

    compute_betas(data_path=str_data_path, force_new_files=False, boo_verbose=boo_verbose)

if str_data_path == "bold":

    compute_bold_files(data_path=str_data_path, boo_standardize=True, verbose=boo_verbose, force_new_files=False)

# --- Split Data -------------------------------------------------------------------------------

lis_subjects_train, lis_subjects_test = train_test_split(lis_subjects, train_size=flo_train_test_ratio, shuffle=True, random_state=42)

int_len_subjects_total = len(lis_subjects)
int_len_subjects_train = len(lis_subjects_train)
int_len_subjects_test = len(lis_subjects_test)

lis_files_train = []
lis_files_test = []

for str_subject in lis_subjects:

    str_subject_betas_path = join(str_data_path, str_subject, "func", "betas")
    lis_subject_beta_files = [x for x in listdir(str_subject_betas_path) if x.endswith(".pt")]

    if str_subject in lis_subjects_train:

        lis_files_train.append(lis_subject_beta_files)
    
    if str_subject in lis_subjects_test:

        lis_files_test.append(lis_subject_beta_files)

lis_files_train = [x for sublist in lis_files_train for x in sublist]
lis_files_test = [x for sublist in lis_files_test for x in sublist]

# --- Build Dataset ---------------------------------------------------------------------------

class Custom_Dataset(Dataset):

    def __init__(self, files):

        self.files = files

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):

        str_file = self.files[idx]
        int_subject_id = str_file.split("_")[0]

        str_file_path = join(str_data_path, f"sub-{int_subject_id}", "func", "betas", str_file)

        file = torch.load(f=str_file_path, weights_only=False)

        tensor = file["image"].unsqueeze(0)
        label = file["label"].float()

        return tensor, label

# --- Setup Dataloaders -----------------------------------------------------------------------

train_dataset = Custom_Dataset(lis_files_train)
test_dataset = Custom_Dataset(lis_files_test)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=int_batch_size, shuffle=True, generator=torch.Generator().manual_seed(int_random_seed), drop_last=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=int_batch_size, shuffle=False)

# --- Setup the CNN ----------------------------------------------------------------------------

def build_model(flo_learning_rate:float = flo_learning_rate, flo_dropout:float=flo_dropout):

    class cla_cnn_3D(Module):        # nn.Module?
        def __init__(self):             # self?
            super().__init__()          # super()?

            self.conv01 = Conv3d(in_channels=1, out_channels=8, kernel_size=(3,3,3), stride=1, padding=1)
            self.conv02 = Conv3d(in_channels=8, out_channels=16, kernel_size=(3,3,3), stride=1, padding=1)
            self.conv03 = Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), stride=1, padding=1)
            self.conv04 = Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=1, padding=1)
            self.conv05 = Conv3d(in_channels=64, out_channels=128, kernel_size=(3,3,3), stride=1, padding=1)
            self.conv06 = Conv3d(in_channels=128, out_channels=256, kernel_size=(3,3,3), stride=1, padding=1)

            self.drop01 = Dropout3d(p=flo_dropout)

            self.batch01 = BatchNorm3d(num_features=8)           # Batchnnorm??
            self.batch02 = BatchNorm3d(num_features=16)
            self.batch03 = BatchNorm3d(num_features=32)
            self.batch04 = BatchNorm3d(num_features=64)
            self.batch05 = BatchNorm3d(num_features=128)
            self.batch06 = BatchNorm3d(num_features=256)

            self.fc01 = LazyLinear(out_features=256)
            self.fc02 = Linear(in_features=256, out_features=1)
        
        def forward(self, x):               # make into dictionary?

            kernel_size = (2,2,2)           
            
            x = self.conv01(x)
            x = self.batch01(x)                                     # why this order?
            x = relu(input=x)
            x = max_pool3d(input=x, kernel_size=kernel_size)      # max-pool vs mean-pool?
            x = self.drop01(x)

            x = self.conv02(x)
            x = self.batch02(x)
            x = relu(input=x)
            x = max_pool3d(input=x, kernel_size=kernel_size)
            x = self.drop01(x)

            x = self.conv03(x)
            x = self.batch03(x)
            x = relu(input=x)
            x = max_pool3d(input=x, kernel_size=kernel_size)
            x = self.drop01(x)

            x = self.conv04(x)
            x = self.batch04(x)
            x = relu(input=x)
            x = max_pool3d(input=x, kernel_size=kernel_size)
            x = self.drop01(x)

            x = self.conv05(x)
            x = self.batch05(x)
            x = relu(input=x)
            x = max_pool3d(input=x, kernel_size=kernel_size)
            x = self.drop01(x)

            x = self.conv06(x)
            x = self.batch06(x)
            x = relu(input=x)
            x = max_pool3d(input=x, kernel_size=kernel_size)
            x = self.drop01(x)

            x = torch.flatten(x, 1)         # ??
            x = relu(self.fc01(x))
            x = self.fc02(x)
            
            return x

    model = cla_cnn_3D().to(device=device)
    loss_func = BCEWithLogitsLoss()          # BCELoss vs Crossentropyloss vs. other?
    optimizer = torch.optim.Adam(params=model.parameters(), lr=flo_learning_rate)       # Adam vs others

    return model, loss_func, optimizer, flo_learning_rate

cnn_model, loss_func, optimizer, flo_learning_rate = build_model(flo_learning_rate=flo_learning_rate)

# ---------------------------------------------------------------------------------------------

def train_model(model, loss_func, optimizer, int_n_epochs:int=int_n_epochs, boo_create_plots:bool = True):

    ten_train_loss = torch.zeros(int_n_epochs)
    ten_train_acc = torch.zeros(int_n_epochs)

    ten_test_loss = torch.zeros(int_n_epochs)
    ten_test_acc = torch.zeros(int_n_epochs)

    int_n_batches = int(ceil(len(train_dataset.files) / train_dataloader.batch_size))

    for int_epoch in (range(int_n_epochs)):

        flo_epoch_begin = time()
        flo_train_begin = time()

        int_n_batch = 1

        lis_train_batch_loss = []
        lis_train_batch_acc = []
        lis_train_batch_precision = []
        lis_train_batch_recall = []
        lis_train_batch_f1 = []

        model.train()         # train vs test??

        for x,y in train_dataloader:

            x = x.to(device=device)

            y = y.to(device=device)
            y = y.reshape(12,1)

            y_hat = model(x)
            loss = loss_func(y_hat, y)

            optimizer.zero_grad()   # ??
            loss.backward()    # ??
            optimizer.step()  # ??

            pred = (y_hat > 0).long()
            matches = (pred == y).long()
            acc = matches.float().mean()

            lis_train_batch_loss.append(loss.item())
            lis_train_batch_acc.append(acc.item())

            str_epoch_progress_bar = progress_bar(int_n_train_batch=int_n_batch, int_n_train_batches=int_n_batches, int_progress_bar_size=3, int_format_width=4)
            print(f"Epoch: {int_epoch + 1}/{int_n_epochs}, Batch: {f"{int_n_batch}/{int_n_batches}":>5} {str_epoch_progress_bar}", end="\r")

            int_n_batch +=1
        
        print("")
        print("")

        ten_train_loss[int_epoch] = mean(lis_train_batch_loss)
        ten_train_acc[int_epoch] = mean(lis_train_batch_acc)

        flo_accuracy_train = round(a=ten_train_acc[int_epoch], decimals=2)*100

        print(f"    Train accuracy: {f"{flo_accuracy_train:.2f}":>5}%")

        flo_train_end = time()

        # --- Testing -----------------------------------------------------------------------------
        
        flo_test_begin = time()
        
        lis_test_batch_loss = []
        lis_test_batch_acc = []
        lis_test_batch_precision = []
        lis_test_batch_recall = []
        lis_test_batch_f1 = []

        model.eval()

        with torch.no_grad():

            for x,y in test_dataloader:

                x = x.to(device)

                y = y.reshape(12,1).float()
                y = y.to(device)

                y_hat = model(x)
                loss = loss_func(y_hat, y)

                pred = (y_hat > 0).long()
                matches = (pred == y).long()
                acc = matches.float().mean()

                lis_test_batch_loss.append(loss.item())
                lis_test_batch_acc.append(acc.item())
            
            ten_test_loss[int_epoch] = mean(lis_test_batch_loss)
            ten_test_acc[int_epoch] = mean(lis_test_batch_acc)

        flo_accuracy_test = round(a=ten_test_acc[int_epoch], decimals=2)*100

        print(f"    Test accuracy: {f"{flo_accuracy_test:.2f}":>6}%")
        print("")

        flo_test_end = time()
        flo_epoch_end = time()

        # --- Logs ------------------------------------------------------------------------------------

        flo_time_epoch = round(a=flo_epoch_end - flo_epoch_begin, decimals=2)
        flo_time_training = round(a=flo_train_end - flo_train_begin, decimals=2)
        flo_time_testing = round(a=flo_test_end - flo_test_begin, decimals=2)

        mem = virtual_memory()
        flo_mem_used = round(mem.used / (1024**3), decimals=2)
        flo_mem_used_percemt = mem.percent
        flo_mem_available = round(mem.available / (1024**3), decimals=2)
        flo_mem_total = round(mem.total / (1024**3), decimals=2)

        flo_cuda_allocated = round(torch.cuda.memory_allocated() / (1024**3), decimals=2)
        flo_cuda_reserved = round(torch.cuda.memory_reserved() / (1024**3), decimals=2)
        flo_cuda_free = round(torch.cuda.mem_get_info()[0] / (1024**3), decimals=2)
        flo_cuda_total = round(torch.cuda.mem_get_info()[1] / (1024**3), decimals=2)


        dic_log_epochs[f"Epoch_{int_epoch+1}"] = {"Accuracy_Train":flo_accuracy_train.item(),
                                                "Accuracy_Test":flo_accuracy_test.item(),
                                                #"Precision_Train":flo_precision_train,
                                                #"Precision_Test":flo_precision_test,
                                                #"Recall_Train":flo_recall_train,
                                                #"Recall_Test":flo_recall_test,
                                                #"F1-Score_Train":flo_f1_train,
                                                #"F1-Score_Test":flo_f1_test,       #TODO add metrics
                                                #"Loss_Train":flo_loss_train,
                                                #"Loss_Test":flo_loss_testm,
                                                "Time_Epoch":flo_time_epoch,
                                                "Time_Train":flo_time_training,
                                                "Time_Test":flo_time_testing,
                                                "RAM_Used":flo_mem_used,
                                                "RAM_Used_Percent":flo_mem_used_percemt,
                                                "RAM_Available":flo_mem_available,
                                                "RAM_Total":flo_mem_total,
                                                "GPU_Allocated":flo_cuda_allocated,
                                                "GPU_Reserved":flo_cuda_reserved,
                                                "GPU_Free":flo_cuda_free,
                                                "GPU_Total":flo_cuda_total}

                                                #TODO check if memory is correct

    #str_cpu, str_cpu_driver, str_gpu, str_gpu_driver, str_ram, str_ram_driver, str_mainboard, str_mainboard_driver = get_hardware()
    #TODO add devices + drivers

    dic_log_run = {"Run-ID":int_run_id,
                "Date":str_date,
                "Time":str_time,
                "Epochs":int_n_epochs, 
                "Learning_Rate":flo_learning_rate,
                "Loss_Function":loss_func,
                "Optimizer":optimizer,
                "Random_Seed":int_random_seed,
                "Batch_Size":int_batch_size,
                "Dropout":flo_dropout,
                "Train_Test_Ratio":flo_train_test_ratio,
                "Subject_Number_Total":int_len_subjects_total,
                "Subject_Number_Training":int_len_subjects_train,
                "Subject_Number_Test":int_len_subjects_test,
                #"CPU":str_cpu,
                #"GPU":str_gpu,
                #"RAM":str_ram,
                #"Mainboard":str_mainboard,
                #"CPU_Driver":str_cpu_driver,
                #"GPU_Driver":str_gpu_driver,
                #"RAM_Driver":str_ram_driver,
                #"Mainboard_Driver":str_mainboard_driver,
                "Python_Version":python_version(),
                "Numpy_Version":np_version,
                "Pandas_Version":pd_version,
                "Sklearn_Version":skl_version,
                "Nibabel_Version":nib_version,
                "Nilearn_Version":nil_version,
                "Torch_Version":tor_version,
                "Data_Source":"https://balsa.wustl.edu/project?project=HCP_YA",
                "Data_Release_Date":"Aug 11, 2025",
                "GitHub":"https://github.com/dman317/Master_Thesis.git"}

    dic_log_subjects = {"Subjects_Train":lis_subjects_train,
                        "Subjects_Test":lis_subjects_test}
    
    # --- Plotting --------------------------------------------------------------------------------

    if boo_create_plots:

        fig, ax = subplots(ncols=2, nrows=1, figsize=(10,5))

        ax[0].plot(ten_train_loss, label="Loss Train")
        ax[0].plot(ten_test_loss, label="Loss Test")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].grid(linewidth=0.2)
        ax[0].legend()

        ax[1].plot(ten_train_acc*100, label="Accuracy Train")
        ax[1].plot(ten_test_acc*100, label="Accuracy Test")
        ax[1].set_ylabel("Accuracy in %")
        ax[1].set_ylim([0,100])
        ax[1].set_yticks(ticks=arange(0,110, 10))
        ax[1].set_xlabel("Epochs")
        ax[1].axhline(y=50, color="black", linestyle="--", label = "Baseline", linewidth=1)
        ax[1].grid(linewidth=0.2)
        ax[1].legend()

        show()

    return model, dic_log_epochs, dic_log_run, dic_log_subjects

# ---------------------------------------------------------------------------------------------

model, dic_log_epochs, dic_log_run, dic_log_subjects = train_model(model=cnn_model, loss_func=loss_func, optimizer=optimizer, int_n_epochs=int_n_epochs, boo_create_plots=True)

# ---------------------------------------------------------------------------------------------

handle_logs(str_data_path=str_data_path, dic_log_epochs=dic_log_epochs, dic_log_run=dic_log_run, dic_log_subjects=dic_log_subjects)

# %%

#CNN(str_data_type="betas",
#    str_data_path="/home/daniel/Desktop/Python/Data/MRI/HCP_AWS",
#    int_n_epochs=25,
#    flo_train_test_ratio=0.8,
#    int_batch_size=12,
#    flo_learning_rate=0.001,
#    flo_dropout=0.2,
#    int_random_seed=42)

# %%

#(net.state_dict())

x = next(iter(train_dataloader))
x[0].shape
x[1].shape