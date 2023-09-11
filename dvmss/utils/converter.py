import numpy as np
import pandas as pd


def pred_data_to_df(pred_d):
    df = pd.DataFrame()
    df["flux_g_x"] = pred_d["vector_receiver"]["sensor_data"][:, 0]
    df["flux_g_y"] = pred_d["vector_receiver"]["sensor_data"][:, 1]
    df["flux_g_z"] = pred_d["vector_receiver"]["sensor_data"][:, 2]
    df["flux_g_perm_x"] = pred_d["vector_receiver"]["perm"][:, 0]
    df["flux_g_perm_y"] = pred_d["vector_receiver"]["perm"][:, 1]
    df["flux_g_perm_z"] = pred_d["vector_receiver"]["perm"][:, 2]
    df["flux_g_induced_x"] = pred_d["vector_receiver"]["induced"][:, 0]
    df["flux_g_induced_y"] = pred_d["vector_receiver"]["induced"][:, 1]
    df["flux_g_induced_z"] = pred_d["vector_receiver"]["induced"][:, 2]
    df["flux_g_interf_x"] = pred_d["vector_receiver"]["interf"][:, 0]
    df["flux_g_interf_y"] = pred_d["vector_receiver"]["interf"][:, 1]
    df["flux_g_interf_z"] = pred_d["vector_receiver"]["interf"][:, 2]
    df["flux_g_gt_x"] = pred_d["vector_receiver"]["ground_true"][:, 0]
    df["flux_g_gt_y"] = pred_d["vector_receiver"]["ground_true"][:, 1]
    df["flux_g_gt_z"] = pred_d["vector_receiver"]["ground_true"][:, 2]
    df["mag_10_uc"] = pred_d["scalar_receiver"]["sensor_data"]
    df["mag_10_uc_perm"] = pred_d["scalar_receiver"]["perm"]
    df["mag_10_uc_induced"] = pred_d["scalar_receiver"]["induced"]
    df["mag_10_uc_interf"] = pred_d["scalar_receiver"]["interf"]
    df["mag_10_uc_gt"] = pred_d["scalar_receiver"]["ground_true"]
    return df
