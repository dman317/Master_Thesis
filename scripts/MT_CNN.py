
# %% --- Load Modules -----------------------------------------------------------------------------

from matplotlib.pyplot import show, subplots
from nibabel import __version__ as nib_version
from nilearn import __version__ as nil_version
from nilearn.image import load_img
from nilearn.masking import unmask
from numpy import arange, ceil, mean, load, round, __version__ as np_version
from pandas import __version__ as pd_version
from platform import python_version
from psutil import virtual_memory
from sklearn import __version__ as skl_version
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import time
from datetime import datetime
import torch
from torch import argmax, float, long, tensor, __version__ as tor_version, flatten
from torch.cuda import is_available
from torch.nn import AdaptiveAvgPool3d, CrossEntropyLoss, BatchNorm3d, ReLU, Conv3d, Dropout3d, LazyLinear, Linear, Softmax, Module, Sequential, Flatten
from torch.nn.functional import max_pool3d, relu
from torch.utils.data import DataLoader, Dataset

from _MT_functions import get_arguments, get_classes_training, get_subjects, get_split_files, get_settings_summary, handle_logs, progress_bar

device = torch.device("cuda" if is_available() else "cpu")


# %% --- Handle Arguments -------------------------------------------------------------------------

args = get_arguments()
#args.int_batch_size_train = 1
#args.lis_tasks_train = ["EMOTION"]

# TODO make sure int_batch_size=12 its correct
# TODO investigate test > train issue
# TODO make it one big function with all the parameters in one function definition
# TODO make it work for bold files
# TODO save weights in each epoch in seperate folder
# TODO check grammar of all prints & and decide what is boo_verbose and what not

# %% ----------------------------------------------------------------------------------------------

lis_subjects = get_subjects(pat_data=args.pat_data)
int_n_subjects_total = len(lis_subjects)

img_mask = load_img(args.pat_data/"task-ALL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz")

lis_files_train, lis_files_test, lis_subjects_train, lis_subjects_test = get_split_files()

lis_classes_train, lis_classes_coded_train = get_classes_training()

dic_log_epochs = {}
dic_log_run = {}
dic_log_subjects = {}

pat_log_path = args.pat_data.parent/"logs"
#lis_log_files = [x for x in pat_log_path.iterdir() if x.name.endswith("run.csv")]
#int_run_id = len(lis_log_files) + 1

str_date = datetime.now().strftime("%d.%m.%Y")
str_time = datetime.now().strftime("%H:%M:%S")

# %% --- Build Dataset ---------------------------------------------------------------------------

class Custom_Dataset(Dataset):

    def __init__(self, files):

        self.files = files

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):

        npz_data = load(self.files[idx])
        img_file = unmask(X=npz_data["data"], mask_img=img_mask)

        ten_data = tensor(img_file.get_fdata(), dtype=float)
        ten_data = ten_data.permute(3,0,1,2)
        #ten_data = ten_data.unsqueeze(0)

        labels = tensor(npz_data["labels"], dtype=long)

        return ten_data, labels

# %% --- Setup Dataloaders -----------------------------------------------------------------------

train_dataset = Custom_Dataset(lis_files_train)
test_dataset = Custom_Dataset(lis_files_test)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.int_batch_size_train, shuffle=True, generator=torch.Generator().manual_seed(args.int_seed), drop_last=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.int_batch_size_test, shuffle=False)

# %% --- Setup the CNN ----------------------------------------------------------------------------

class cla_cnn_3D(Module):
    def __init__(self):
        super().__init__()

        self.conv01 = Conv3d(in_channels=1,   out_channels=8,   kernel_size=(3,3,3), stride=1, padding=1)
        self.conv02 = Conv3d(in_channels=8,   out_channels=8,   kernel_size=(3,3,3), stride=1, padding=1)
        self.conv03 = Conv3d(in_channels=8,   out_channels=8,   kernel_size=(3,3,3), stride=2, padding=1)
        self.conv04 = Conv3d(in_channels=8,   out_channels=8,   kernel_size=(3,3,3), stride=1, padding=1)
        self.conv05 = Conv3d(in_channels=8,   out_channels=16,  kernel_size=(3,3,3), stride=2, padding=1)
        self.conv06 = Conv3d(in_channels=16,  out_channels=16,  kernel_size=(3,3,3), stride=1, padding=1)
        self.conv07 = Conv3d(in_channels=16,  out_channels=32,  kernel_size=(3,3,3), stride=2, padding=1)
        self.conv08 = Conv3d(in_channels=32,  out_channels=32,  kernel_size=(3,3,3), stride=1, padding=1)
        self.conv09 = Conv3d(in_channels=32,  out_channels=64,  kernel_size=(3,3,3), stride=2, padding=1)
        self.conv10 = Conv3d(in_channels=64,  out_channels=64,  kernel_size=(3,3,3), stride=1, padding=1)
        self.conv11 = Conv3d(in_channels=64,  out_channels=128, kernel_size=(3,3,3), stride=2, padding=1)
        self.conv12 = Conv3d(in_channels=128, out_channels=128, kernel_size=(3,3,3), stride=1, padding=1)

        self.dropout = Dropout3d(0.2)

        self.batchnorm8 =   BatchNorm3d(num_features=8)
        self.batchnorm16 =  BatchNorm3d(num_features=16)
        self.batchnorm32 =  BatchNorm3d(num_features=32)
        self.batchnorm64 =  BatchNorm3d(num_features=64)
        self.batchnorm128 = BatchNorm3d(num_features=128)

        self.softmmax = Softmax(dim=1)

        self.adaptiveavgepool3d = AdaptiveAvgPool3d(output_size=(1,1,1))

        self.fc01 = LazyLinear(out_features=128)
        self.fc02 = Linear(in_features=128, out_features=len(lis_classes_train))
            
    def forward(self, x):

        x = self.conv01(x)
        x = relu(input=x)
        x = self.dropout(x)
        x = self.batchnorm8(x)

        x = self.conv02(x)
        x = relu(input=x)
        x = self.dropout(x)
        x = self.batchnorm8(x)

        x = self.conv03(x)
        x = relu(input=x)
        x = self.dropout(x)
        x = self.batchnorm8(x)

        x = self.conv04(x)
        x = relu(input=x)
        x = self.dropout(x)
        x = self.batchnorm8(x)

        x = self.conv05(x)
        x = relu(input=x)
        x = self.dropout(x)
        x = self.batchnorm16(x)

        x = self.conv06(x)
        x = relu(input=x)
        x = self.dropout(x)
        x = self.batchnorm16(x)

        x = self.adaptiveavgepool3d(x)

        x = flatten(input=x, start_dim= 1)
        x = relu(self.fc01(x))
        x = self.fc02(x)

        return x

model = cla_cnn_3D().to(device=device)
loss_func = CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.flo_learning_rate)       # Adam vs others

# %% ----------------------------------------------------------------------------------------------

get_settings_summary(args=args, int_n_subjects_total=int_n_subjects_total, lis_classes_train=lis_classes_train)

print(" --- Begin Training ------------------------------------------------------------------------")
print("")

ten_train_loss = torch.zeros(args.int_epochs)
ten_train_acc = torch.zeros(args.int_epochs)

ten_test_loss = torch.zeros(args.int_epochs)
ten_test_acc = torch.zeros(args.int_epochs)

int_n_batches = int(ceil(len(train_dataset.files) / train_dataloader.batch_size))

for int_epoch in (range(args.int_epochs)):

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

        x = x.permute(1,0,2,3,4)
        x = x.to(device=device)
        y = y.long().reshape(-1).to(device=device)

        y_hat = model(x)
        loss = loss_func(y_hat, y)

        optimizer.zero_grad()   # ??
        loss.backward()    # ??
        optimizer.step()  # ??

        pred = argmax(input=y_hat, dim=1)
        matches = (pred == y).float()

        acc = matches.mean()

        lis_train_batch_loss.append(loss.item())
        lis_train_batch_acc.append(acc.item())

        str_epoch_progress_bar = progress_bar(int_n_train_batch=int_n_batch, int_n_train_batches=int_n_batches, int_progress_bar_size=3, int_format_width=4)
        print(f"Epoch: {int_epoch + 1}/{args.int_epochs}, Batch: {f"{int_n_batch}/{int_n_batches}":>5} {str_epoch_progress_bar}", end="\r")

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

            x = x.permute(1,0,2,3,4)
            x = x.to(device)

            y = y.long().reshape(-1).to(device=device)

            y_hat = model(x)
            loss = loss_func(y_hat, y)

            pred = argmax(input=y_hat, dim=1)
            matches = (pred == y).float()

            acc = matches.mean()

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

dic_log_run = {"Run-ID":"",
            "Date":str_date,
            "Time":str_time,
            "Epochs":args.int_epochs, 
            "Learning_Rate":args.flo_learning_rate,
            "Loss_Function":loss_func,
            "Optimizer":optimizer,
            "Random_Seed":args.int_seed,
            "Batch_Size":args.int_batch_size_train,
            "Dropout":args.flo_dropout,
            "Subject_Number_Total":int_n_subjects_total,
            "Subject_Number_Training":args.int_n_subjects_train,
            "Subject_Number_Test":args.int_n_subjects_test,
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
    
# ---------------------------------------------------------------------------------------------

handle_logs(str_data_path=args.pat_data.parent, dic_log_epochs=dic_log_epochs, dic_log_run=dic_log_run, dic_log_subjects=dic_log_subjects)

# %%

#CNN(str_data_type="betas",
#    str_data_path="/home/daniel/Desktop/Python/Data/MRI/HCP_AWS",
#    int_n_epochs=25,
#    flo_train_test_ratio=0.8,
#    int_batch_size=12,
#    flo_learning_rate=0.001,
#    flo_dropout=0.2,
#    int_random_seed=42)





