
# %%

from nibabel import save as save_img
from nilearn.image import load_img
from nilearn.masking import intersect_masks
from pathlib import Path

from _MT_functions import get_runs, get_subjects, get_tasks

pat_data = Path.cwd().parent/"data_raw"
boo_force_new_files = False
boo_move_mask_files = True

lis_pat_subjects = get_subjects(pat_data=pat_data)
lis_tasks = get_tasks()
lis_runs = get_runs()
lis_task_masks_intersect = []
pat_mask_all = pat_data/f"task-ALL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz"
str_mask_all = pat_mask_all.name

print(f"Path: {pat_data}")
print("")

# %%

for str_task in lis_tasks:

    str_mask_name = f"task-{str_task}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz"
    pat_mask_file = pat_data/str_mask_name

    if not pat_mask_file.is_file() or boo_force_new_files:

        print(f"{f"{str_mask_name:<88}"}\033[91mFILE MISSING\033[0m", end="\r")

        lis_task_masks = []

        for pat_subject in lis_pat_subjects:

            str_subject = pat_subject.name

            for str_run in lis_runs:

                img_subject_run_mask = load_img(pat_data/str_subject/"func"/f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz")

                lis_task_masks.append(img_subject_run_mask)

        img_mask_intersect = intersect_masks(mask_imgs=lis_task_masks, threshold=1)

        lis_task_masks.append(img_mask_intersect)
        
        save_img(img=img_mask_intersect, filename=pat_mask_file)

        print(f"{f"{str_mask_name:<88}"}\033[92mfile created\033[0m")

    else:

        print(f"{f"{str_mask_name:<88}"}\033[92mfile found\033[0m")

print("")

if not pat_mask_all.is_file():

    print(f"{f"{str_mask_all:<88}"}\033[91mFILE MISSING\033[0m", end="\r")

    lis_task_masks_intersect = [mask for mask in pat_data.iterdir() if mask.name.endswith("intersect.nii.gz")]

    img_all_masks_intersect = intersect_masks(lis_task_masks_intersect, threshold=1)

    save_img(img=img_all_masks_intersect, filename=pat_mask_all)

    print(f"{f"{str_mask_all:<88}"}\033[92mfile created\033[0m")

else:

    print(f"{f"{str_mask_all:<88}"}\033[92mfile found\033[0m")

# %%

if boo_move_mask_files:

    for pat_subject in lis_pat_subjects:

        pat_subject_func = pat_subject/"func"
        pat_subject_func_masks = pat_subject_func/"task_mask_files"

        if not pat_subject_func_masks.is_dir():

            pat_subject_func_masks.mkdir()

        for pat_file in pat_subject_func.iterdir():

            str_file = pat_file.name

            if str_file.endswith("brain_mask.nii.gz"):

                pat_file.rename(target=pat_subject_func_masks/str_file)
