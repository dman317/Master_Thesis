
# %% --- Load Modules -----------------------------------------------------------------------------

from argparse import ArgumentParser
from nibabel import save as save_img
from nilearn.image import load_img
from nilearn.masking import intersect_masks
from pathlib import Path

from _MT_functions import get_runs, get_subjects, get_tasks

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = ArgumentParser(description="???")

ap.add_argument("--data", required=False, default=Path(__file__).parent.parent/"data")
ap.add_argument("--force_new_files", required=False, default=False)

args = ap.parse_args()

pat_data = args.data
boo_force_new_files = args.force_new_files

# %% --- Load Data --------------------------------------------------------------------------------

lis_pat_subjects = get_subjects(pat_data=pat_data)
lis_tasks = get_tasks()
lis_runs = get_runs()

lis_task_masks_intersect = []

str_mask_all = "task-ALL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz"
pat_mask_all = pat_data/str_mask_all

# %% --- Generate Intersect Masks -----------------------------------------------------------------

for str_task in lis_tasks:

    str_mask_name = f"task-{str_task}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask_intersect.nii.gz"
    pat_mask_file = pat_data/str_mask_name

    if not pat_mask_file.is_file() or boo_force_new_files:

        lis_task_masks = []

        for pat_subject in lis_pat_subjects:

            str_subject = pat_subject.name

            for str_run in lis_runs:

                img_subject_run_mask = load_img(pat_data/str_subject/"func"/"task_mask_files/"f"{str_subject}_task-{str_task}_run-{str_run}_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz")

                lis_task_masks.append(img_subject_run_mask)

        img_mask_intersect = intersect_masks(mask_imgs=lis_task_masks, threshold=1)
        
        save_img(img=img_mask_intersect, filename=pat_mask_file)

        print(f"{f"{str_mask_name:<88}"}\033[92mfile created\033[0m")

    else:

        print(f"{f"{str_mask_name:<88}"}\033[92mfile found\033[0m")

print("")

if not pat_mask_all.is_file():

    print(f"{f"{str_mask_all:<88}"}\033[93mgenerating file...\033[0m", end="\r")

    lis_task_masks_intersect = [mask for mask in pat_data.iterdir() if mask.name.endswith("intersect.nii.gz")]

    img_all_masks_intersect = intersect_masks(lis_task_masks_intersect, threshold=1)

    save_img(img=img_all_masks_intersect, filename=pat_mask_all)

    print(f"{f"{str_mask_all:<88}"}\033[92mfile created\033[0m")

else:

    print(f"{f"{str_mask_all:<88}"}\033[92mfile found\033[0m")
