# %%

#TODO write text for 'help' and explain a bit, call the expected data location

import argparse
from numpy import unique
from pathlib import Path
from shutil import rmtree

from _MT_functions import get_longest_file_name

# %% --- Handle Arguments -------------------------------------------------------------------------

ap = argparse.ArgumentParser(description="???") #TODO
ap.add_argument("--data", required=False, default=Path("../data/"), help=f"???")
ap.add_argument("--remove", required=False, default=False, help="???")

args = ap.parse_args()

pat_data = Path(str(args.data))
bool_remove = eval(str(args.remove))

# %% ---

lis_subjects = [pat for pat in pat_data.iterdir() if pat.name.startswith("sub")]

lis_files = ['task-WM_run-LR_desc-EV_0bk_body.txt',
                'task-RELATIONAL_run-LR_desc-EV_relation.txt',
                'task-LANGUAGE_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-RELATIONAL_run-RL_desc-EV_relation.txt',
                'task-LANGUAGE_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-WM_run-RL_desc-EV_0bk_cor.txt',
                'task-EMOTION_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-WM_run-LR_desc-EV_2bk_body.txt',
                'task-GAMBLING_run-RL_desc-EV_win.txt',
                'task-WM_run-LR_desc-EV_0bk_faces.txt',
                'task-GAMBLING_run-RL_desc-EV_loss.txt',
                'task-RELATIONAL_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-GAMBLING_run-LR_desc-EV_neut_event.txt',
                'task-MOTOR_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-RELATIONAL_run-LR_desc-EV_match.txt',
                'task-WM_run-RL_desc-EV_all_bk_err.txt',
                'task-WM_run-LR_desc-EV_2bk_cor.txt',
                'task-WM_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-MOTOR_run-RL_desc-EV_cue.txt',
                'task-WM_run-LR_desc-EV_2bk_nlr.txt',
                'task-SOCIAL_run-LR_desc-EV_mental_resp.txt',
                'task-EMOTION_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-WM_run-LR_desc-EV_0bk_places.txt',
                'task-GAMBLING_run-LR_desc-EV_win_event.txt',
                'task-GAMBLING_run-LR_desc-EV_win.txt',
                'task-MOTOR_run-LR_desc-EV_lh.txt',
                'task-WM_run-RL_desc-EV_2bk_places.txt',
                'task-GAMBLING_run-RL_desc-EV_neut_event.txt',
                'task-EMOTION_run-RL_desc-EV_fear.txt',
                'task-RELATIONAL_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-WM_run-RL_desc-EV_0bk_err.txt',
                'task-WM_run-RL_desc-EV_2bk_body.txt',
                'task-SOCIAL_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-MOTOR_run-LR_desc-EV_cue.txt',
                'task-WM_run-RL_desc-EV_0bk_nlr.txt',
                'task-WM_run-RL_desc-EV_0bk_tools.txt',
                'task-WM_run-LR_desc-EV_2bk_err.txt',
                'task-WM_run-RL_desc-EV_2bk_faces.txt',
                'task-MOTOR_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-MOTOR_run-LR_desc-EV_lf.txt',
                'task-WM_run-RL_desc-EV_0bk_faces.txt',
                'task-SOCIAL_run-LR_desc-EV_mental.txt',
                'task-LANGUAGE_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-WM_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-EMOTION_run-LR_desc-EV_fear.txt',
                'task-WM_run-RL_desc-EV_0bk_places.txt',
                'task-GAMBLING_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-WM_run-LR_desc-EV_0bk_err.txt',
                'task-MOTOR_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-MOTOR_run-RL_desc-EV_lh.txt',
                'task-LANGUAGE_run-LR_desc-EV_story.txt',
                'task-WM_run-LR_desc-EV_0bk_tools.txt',
                'task-RELATIONAL_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-WM_run-RL_desc-EV_all_bk_cor.txt',
                'task-WM_run-RL_desc-EV_0bk_body.txt',
                'task-WM_run-LR_desc-EV_2bk_places.txt',
                'task-EMOTION_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-EMOTION_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-SOCIAL_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-MOTOR_run-LR_desc-EV_rh.txt',
                'task-WM_run-RL_desc-EV_2bk_err.txt',
                'task-WM_run-LR_desc-EV_0bk_cor.txt',
                'task-WM_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-MOTOR_run-RL_desc-EV_rh.txt',
                'task-GAMBLING_run-RL_desc-EV_win_event.txt',
                'task-WM_run-LR_desc-EV_2bk_faces.txt',
                'task-MOTOR_run-LR_desc-EV_rf.txt',
                'task-GAMBLING_run-LR_desc-EV_loss_event.txt',
                'task-SOCIAL_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-WM_run-LR_desc-EV_all_bk_err.txt',
                'task-WM_run-RL_desc-EV_2bk_cor.txt',
                'task-MOTOR_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-WM_run-RL_desc-EV_2bk_nlr.txt',
                'task-SOCIAL_run-RL_desc-EV_rnd.txt',
                'task-EMOTION_run-RL_desc-EV_neut.txt',
                'task-SOCIAL_run-RL_desc-EV_mental.txt',
                'task-WM_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-MOTOR_run-RL_desc-EV_lf.txt',
                'task-RELATIONAL_run-RL_desc-EV_match.txt',
                'task-LANGUAGE_run-RL_desc-EV_math.txt',
                'task-SOCIAL_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-LANGUAGE_run-LR_desc-EV_math.txt',
                'task-MOTOR_run-LR_desc-EV_t.txt',
                'task-RELATIONAL_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-WM_run-LR_desc-EV_all_bk_cor.txt',
                'task-MOTOR_run-RL_desc-EV_t.txt',
                'task-LANGUAGE_run-RL_desc-EV_story.txt',
                'task-WM_run-LR_desc-EV_0bk_nlr.txt',
                'task-SOCIAL_run-RL_desc-EV_other_resp.txt',
                'task-GAMBLING_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
                'task-GAMBLING_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-WM_run-RL_desc-EV_2bk_tools.txt',
                'task-SOCIAL_run-RL_desc-EV_mental_resp.txt',
                'task-GAMBLING_run-LR_desc-EV_loss.txt',
                'task-MOTOR_run-RL_desc-EV_rf.txt',
                'task-SOCIAL_run-LR_desc-EV_other_resp.txt',
                'task-GAMBLING_run-RL_desc-EV_loss_event.txt',
                'task-SOCIAL_run-LR_desc-EV_rnd.txt',
                'task-GAMBLING_run-RL_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz',
                'task-WM_run-LR_desc-EV_2bk_tools.txt',
                'task-EMOTION_run-LR_desc-EV_neut.txt',
                'task-LANGUAGE_run-LR_space-MNI152NLin6Asym_res-2_desc-preproc_brain_mask.nii.gz']

str_longest_file_name, int_len_longest_file_name = get_longest_file_name(lis_files=lis_files)

lis_subjects_incomplete = []

for int_subject_i, pat_subject in enumerate(lis_subjects, start=1):

    str_subject = pat_subject.name

    int_format_width = len(str_subject) + int_len_longest_file_name + 4

    print(f"Subject: {str_subject} ({int_subject_i}/{len(lis_subjects)})")

    for str_file in lis_files:

        str_file = f"{str_subject}_{str_file}"

        if (pat_subject/"func"/str_file).exists():

            print(f"{str_file:<{int_format_width}} \33[92mfile found\033[0m")

        else:

            print(f"{str_file:<{int_format_width}} \033[91mFILE MISSING\033[0m")

            lis_subjects_incomplete.append(pat_subject)

    print("")

lis_unique_subjects_incomplete = unique(lis_subjects_incomplete)

print(f"Subjects with incomplete data: {len(lis_unique_subjects_incomplete)}")
print(f"Subjects: {lis_unique_subjects_incomplete}")
print("")


# %%

if bool_remove and lis_subjects_incomplete:

    str_input = input("Are you sure you want to delete all subjects with incomplete data? This can not be undone and you have to download their data again, if you need it later. (y/n)")

    bool_valid_input = str_input in ["y", "n"]

    while not bool_valid_input:

        print("Invalid input! Use 'y' for 'yes' or 'n' for 'no!")

        str_input = input("Are you sure you want to delete all subjects with incomplete data? This can not be undone and you have to download their data again, if you need it later. (y/n)")

        bool_valid_input = str_input in ["y", "n"]

    if str_input == "y":

        for pat_subject in lis_unique_subjects_incomplete:

            rmtree(pat_subject)

        print("")
        print("All subjects with incomplete data have been deleted.")

