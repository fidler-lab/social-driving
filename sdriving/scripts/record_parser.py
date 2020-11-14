from typing import List, Dict

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
    highway_env: bool = False,
    normalized_lane_position: bool = False,
    distance_to_intersection: bool = False,
    time_to_intersection: bool = False,
    remove_no_signal: bool = False,
    fix_communication: bool = False,
    nframe_size: int = 15.0,
):
    df = pd.read_csv(path, index_col=0)

    # Convert to categorical data
    for categorical_labels in ["Episode", "Agent ID", "Traffic Signal"]:
        if categorical_labels in df:
            df[categorical_labels] = df[categorical_labels].astype("category")

    # Fix the Position column
    df["Position"] = [
        np.fromstring(pos[1:-1], sep=", ") for pos in df["Position"]
    ]

    if (
        normalized_lane_position
        or distance_to_intersection
        and not highway_env
    ):
        if normalized_lane_position:
            norm_lanepos = np.zeros(df.shape[0])
            rpockets = np.zeros(df.shape[0])
            normalized_frame = [0] * df.shape[0]
        if distance_to_intersection:
            dist_to_int = np.zeros(df.shape[0])
        for aid in df["Agent ID"].unique():
            prev_ep = 0
            da = df[df["Agent ID"] == aid]
            for idx, row in da.iterrows():
                pos = row["Position"]
                if row["Episode"] != prev_ep:
                    prev_ep = row["Episode"]
                    rpocket = _get_road_pocket(pos, row["Env Width"])
                    mfactor = -1 if rpocket in [0, 3] else 1
                if normalized_lane_position:
                    norm_lanepos[idx] = (
                        pos[(rpocket + 1) % 2] * mfactor * 2 / row["Env Width"]
                    )
                    rpockets[idx] = rpocket
                    if rpocket in [0, 2]:
                        normalized_frame[idx] = [
                            pos[rpocket % 2],
                            pos[(rpocket + 1) % 2]
                            * nframe_size
                            * 2
                            / row["Env Width"],
                        ]
                    else:
                        normalized_frame[idx] = [
                            pos[(rpocket + 1) % 2]
                            * nframe_size
                            * 2
                            / row["Env Width"],
                            pos[rpocket % 2],
                        ]
                if distance_to_intersection:
                    dist_to_int[idx] = (
                        np.abs(pos[rpocket % 2]) - np.abs(row["Env Width"]) / 2
                    )
        if distance_to_intersection:
            df["Distance to Intersection"] = dist_to_int
        if normalized_lane_position:
            df["Normalized Lane Position"] = norm_lanepos
            df["Road Pocket"] = rpocket
            df["Normalized Frame Position"] = normalized_frame

    if time_to_intersection:
        _dfs = []
        for a_id in df["Agent ID"].unique():
            da = df[df["Agent ID"] == a_id]
            time_to_intersection = []
            prev_ep = -1
            not_reached = False
            l = 0
            for _, row in da.iterrows():
                if row["Episode"] > prev_ep:
                    prev_ep = row["Episode"]
                    if l != 0:
                        time_to_intersection.extend(
                            [arrival_t - i for i in range(l)]
                        )
                    arrival_t = -1
                    l = 0
                    not_reached = True
                if row["Distance to Intersection"] < 2.5 and not_reached:
                    arrival_t = row["Time Step"]
                    not_reached = False
                l += 1
            if l != 0:
                time_to_intersection.extend([arrival_t - i for i in range(l)])
            da.loc[:, "Time to Intersection"] = time_to_intersection
            _dfs.append(da)
        df = pd.concat(_dfs)

    if highway_env and normalized_lane_position:
        df["Normalized Lane Position"] = [
            2 * pos[1] / w for pos, w in zip(df["Position"], df["Env Width"])
        ]

    if remove_no_signal:
        df = df[df["Traffic Signal"] != 0.75]
        df["Traffic Signal"] = df[
            "Traffic Signal"
        ].cat.remove_unused_categories()

    if fix_communication:
        df["Communication (Recv)"] = [
            float(x[1:-1]) for x in df["Communication (Recv)"]
        ]
        df["Communication (Send)"] = [
            float(x[1:-1]) for x in df["Communication (Send)"]
        ]

    return df


def merge_dataframes(paths: List[str], tags: Dict[str, list], **kwargs):
    dfs = []
    for i, path in enumerate(paths):
        df = read_dataframe(path, **kwargs)
        for col_name, val_list in tags.items():
            tag = val_list[i]
            df[col_name] = [tag] * df.shape[0]
        dfs.append(df)
    df = pd.concat(dfs)

    df[col_name] = df[col_name].astype("category")

    return df
