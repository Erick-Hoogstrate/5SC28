# Folder overview

- NARX_GP_GS contains the gridsearch for the NARX_GP model. val_pred_NRMSs.npz and val_sim_NRMSs are now loaded in and the file returns the result.
- NARX_GP_model contains the file used to obtain the NARX GP model and the evaluation of the NARX GP model. It has an option to save or load in models.
- data.py and util_fun.py contain supporting files for NARX_GP_GS and NARX_GP_model.
- disc-benchmark-files contain the files required for system identification by the course.