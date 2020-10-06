from typing import List

import numpy as np
import pandas as pd

def _get_road_pocket(pos: np.array, width: float):
    """
    This is not a generalized function. It only computes correct values
    for the 4-way intersection
    """
    lw = width / 2

    if pos[0] >= lw:
        return 0
    if pos[0] <= -lw:
        return 2
    if pos[1] >= lw:
        return 1
    if pos[1] <= -lw:
        return 3
    return -1


def read_dataframe(
    path: str,
    highway_env: str,
    normalized_lane_position: bool = False,
    distance_to_intersection: bool = False,
    remove_no_signal: bool = False,
    fix_communication: bool = False,
):
    df = pd.read_csv(path, index_col=0)
    
    # Convert to categorical data
    for categorical_labels in ["Episode", "Agent ID", "Traffic Signal"]:
        if categorical_labels in df:
            df[categorical_labels] = df[categorical_labels].astype("category")
    
    # Fix the Position column
    df["Position"] = [np.fromstring(pos) for pos in df["Position"]]

    if normalized_lane_position or distance_to_intersection and not highway_env:
        if normalized_lane_position:
            norm_lanepos = np.zeros(df.shape[0])
        if distance_to_intersection:
            dist_to_int = np.zeros(df.shape[0])
        for aid in df["Agent ID"].unique():
            prev_ep = 0
            da = df[df["Agent ID"] == aid]
            for idx, row in da.iterrows():
                pos = row["Position"]
                if row["Episode"] != prev_ep:
                    prev_ep = row["Episode"]
                    rpocket = _get_road_pocket(pos, row["Env Length"], row["Env Width"])
                    mfactor = -1 if rpocket in [0, 3] else 1
                if normalized_lane_position:
                    norm_lanepos[idx] = pos[(rpocket + 1) % 2] * mfactor * 2 / row["Env Width"]
                if distance_to_intersection:
                    dist_to_int[idx] = np.abs(pos[rpocket % 2]) - np.abs(w)
        if distance_to_intersection:
            df["Distance to Intersection"] = dist_to_int
        if normalized_lane_position:
            df["Normalized Lane Position"] = norm_lanepos

    if highway_env and normalized_lane_position:
        df["Normalized Lane Position"] = [
            2 * pos[1] / w for pos, w in zip(df["Position"], df["Env Width"])
        ] 
    
    if remove_no_signal:
        df = df[df["Traffic Signal"] != 0.75]
        df["Traffic Signal"] = df["Traffic Signal"].remove_unused_categories()

    if fix_communication:
        df["Communication (Recv)"] = [float(x[1:-1]) for x in df["Communication (Recv)"]]
        df["Communication (Send)"] = [float(x[1:-1]) for x in df["Communication (Send)"]]

    return df


def merge_datasets(
    paths: List[str],
    tags: List[str],
    column_name: str,
    **kwargs
):
    dfs = []
    for path, tag in zip(paths, tags):
        df = read_dataframe(path)
        df[col_name] = [tag] * df.shape[0]
        dfs.append(df)
    df = pd.concat(dfs)

    df[col_name] = df[col_name].astype("category")

    return df
