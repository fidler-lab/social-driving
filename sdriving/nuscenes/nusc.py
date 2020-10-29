import json
import math
import sys
from collections import deque
from glob import glob
from typing import Optional

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# use the master branch of nuscenes-devkit instead of pip installed version
import torch
from fire import Fire
from nuscenes.map_expansion.arcline_path_utils import discretize_lane
from nuscenes.map_expansion.map_api import NuScenesMap
from sklearn.neighbors import KDTree

from sdriving.nuscenes.utils import (
    get_drivable_area_matrix,
    nuscenes_map_to_line_representation,
)
from sdriving.nuscenes.world import NuscenesWorld
from sdriving.tsim import angle_normalize


def get_nusc_maps(map_folder):
    nusc_maps = {
        map_name: NuScenesMap(dataroot=map_folder, map_name=map_name)
        for map_name in [
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
            "singapore-onenorth",
        ]
    }
    return nusc_maps


def get_road_img(nmap, midx, midy, width, height, resolution):
    dx, bx, (nx, ny) = get_grid(
        [-width / 2, -height / 2, width / 2, height / 2],
        [resolution, resolution],
    )

    layers = ["road_segment", "lane"]
    lmap = get_local_map(nmap, (midx, midy), width, height, layers)

    road_img = np.zeros((nx, ny))
    for layer_name in layers:
        for poly in lmap[layer_name]:
            # draw the lines
            pts = np.round(
                (poly - np.array([[midx, midy]]) - bx[:2] + dx[:2] / 2.0)
                / dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(road_img, [pts], 1.0)

    return road_img, dx, bx, nx, ny


class MapHelper(object):
    def __init__(self, nusc_maps):
        # map_name -> nuscenesMap
        self.nusc_maps = nusc_maps

        self.info, self.trees = self.prepro_closest()

    def prepro_closest(self):
        print("preprocessing maps...")
        info = {}
        trees = {}
        for map_name, nmap in self.nusc_maps.items():
            info[map_name] = {}
            for lane in nmap.lane + nmap.lane_connector:
                my_lane = nmap.arcline_path_3.get(lane["token"], [])
                info[map_name][lane["token"]] = discretize_lane(
                    my_lane, resolution_meters=0.5
                )
                assert len(info[map_name][lane["token"]]) > 0

            data = np.array(
                [pt for lane in info[map_name] for pt in info[map_name][lane]]
            )[:, :2]
            keys = [
                (map_name, lane, pti)
                for lane in info[map_name]
                for pti, pt in enumerate(info[map_name][lane])
            ]
            tree = KDTree(data)

            trees[map_name] = {"tree": tree, "keys": keys}
        print("done!")

        return info, trees

    def closest(self, map_name, x, y):
        distance, index = self.trees[map_name]["tree"].query([[x, y]])
        _, lane_key, pti = self.trees[map_name]["keys"][index[0, 0]]
        pt = self.info[map_name][lane_key][pti]
        return pt, lane_key, pti

    def bfs(self, map_name, src, tgt):
        nmap = self.nusc_maps[map_name]

        queue = deque()
        queue.append(src)
        tree = {}
        tree[src] = "root"

        while queue:
            s = queue.popleft()
            for i in nmap.connectivity[s]["outgoing"]:
                # some ghost lanes to avoid https://github.com/nutonomy/nuscenes-devkit/issues/415
                if i not in nmap.connectivity:
                    continue
                if i not in tree:
                    queue.append(i)
                    tree[i] = s
            if tgt in tree:
                break

        # path exists
        if tgt in tree:
            path = [tgt]
            while path[-1] != src:
                path.append(tree[path[-1]])
            full_path = list(reversed(path))

        # no path exists
        else:
            full_path = None

        return full_path, tree

    def get_lane_path(self, map_name, p0, p1):
        closep0, lane_key0, pti0 = self.closest(map_name, p0[0], p0[1])
        closep1, lane_key1, pti1 = self.closest(map_name, p1[0], p1[1])
        path, tree = self.bfs(map_name, lane_key0, lane_key1)
        if path is None:
            pts = None
        elif len(path) == 1:
            pts = self.info[map_name][lane_key0][pti0 : (pti1 + 1)]
        else:
            pts = self.info[map_name][lane_key0][pti0:]
            for k in path[1:-1]:
                pts.extend(self.info[map_name][k])
            pts.extend(self.info[map_name][lane_key1][: (pti1 + 1)])
        return pts

    def check_in_box(self, pt, center, width, height):
        return (abs(pt[0] - center[0]) < width / 2) and (
            abs(pt[1] - center[1]) < height / 2
        )

    def collect_paths(self, map_name, starts, center, width, height):
        """Given Nx2 start positions, find all paths that leave those positions
        and end somewhere within the box defined by center, width, height.
        """
        nmap = self.nusc_maps[map_name]
        all_paths = {}

        for starti, start in enumerate(starts):
            all_paths[starti] = []
            _, lane_key, pti = self.closest(map_name, start[0], start[1])

            # point-wise BFS
            queue = deque()
            src = (lane_key, pti)
            queue.append(src)
            tree = {src: "root"}
            endpoints = []

            while queue:
                la, pi = queue.popleft()
                cur_len = len(queue)
                if pi + 1 < len(self.info[map_name][la]):
                    newpt = self.info[map_name][la][pi + 1]
                    cand = (la, pi + 1)
                    if cand not in tree and self.check_in_box(
                        newpt, center, width, height
                    ):
                        queue.append(cand)
                        tree[cand] = (la, pi)

                else:
                    for i in nmap.connectivity[la]["outgoing"]:
                        if i not in nmap.connectivity:
                            continue
                        cand = (i, 0)
                        if cand not in tree:
                            queue.append(cand)
                            tree[cand] = (la, pi)

                # we found an "endpoint"
                if len(queue) == cur_len:
                    endpoints.append((la, pi))
            for endpoint in endpoints:
                path = [endpoint]
                while path[-1] != src:
                    path.append(tree[path[-1]])
                full_path = list(reversed(path))
                all_paths[starti].append(
                    [self.info[map_name][la][pi] for la, pi in full_path]
                )

        return all_paths


def get_grid(point_cloud_range, voxel_size):
    lower = np.array(point_cloud_range[: (len(point_cloud_range) // 2)])
    upper = np.array(point_cloud_range[(len(point_cloud_range) // 2) :])

    dx = np.array(voxel_size)
    bx = lower + dx / 2.0
    nx = list(map(int, (upper - lower) / dx))

    return dx, bx, nx


def get_local_map(nmap, center, width, height, layer_names):
    # need to get the map here...
    box_coords = (
        center[0] - width / 2,
        center[1] - height / 2,
        center[0] + width / 2,
        center[1] + height / 2,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(
        box_coords, layer_names=layer_names, mode="intersect"
    )
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == "drivable_area":
                polygon_tokens = poly_record["polygon_tokens"]
            else:
                polygon_tokens = [poly_record["polygon_token"]]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    return polys


def find_center(
    map_folder="/Users/jonahphilion/Downloads/nuScenes-map-expansion-v1.2",
    map_name="boston-seaport",
):
    nusc_maps = get_nusc_maps(map_folder)
    nmap = nusc_maps[map_name]
    pose_lists = nmap.discretize_centerlines(resolution_meters=0.5)

    def onclick(event):
        print(event)

    fig = plt.figure()
    for pose_list in pose_lists:
        if len(pose_list) > 0:
            plt.plot(pose_list[:, 0], pose_list[:, 1])
    ax = plt.gca()
    ax.set_aspect("equal")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


def env_create(
    map_folder="/Users/jonahphilion/Downloads/nuScenes-map-expansion-v1.2",
    map_name="boston-seaport",
    midx=1289.15,
    midy=1049.04,
    width=100.0,  # meters
    height=100.0,  # meters
    resolution=0.3,
):
    mpl.use("TkAgg")

    nusc_maps = get_nusc_maps(map_folder)
    maphelper = MapHelper(
        {k: v for k, v in nusc_maps.items() if k == map_name}
    )
    road_img, dx, bx, nx, ny = get_road_img(
        nusc_maps[map_name], midx, midy, width, height, resolution
    )

    class GUI(object):
        def __init__(self):
            fig = plt.figure(figsize=(7, 7))
            fig.canvas.mpl_connect("button_press_event", self.onclick)
            fig.canvas.mpl_connect("key_press_event", self.onpress)
            self.starts = []
            self.traffic_signals = []
            self.starts_to_traffic_signal = []
            self.ts_mapping = []
            self.signal_num = 0
            self.map_num = -1
            self.last_mapped = -1

        def onclick(self, event):
            print(event)
            if event.button == 1:
                self.starts.append([event.xdata, event.ydata])
                self.render_pts()
                print(self.starts)
            elif event.button == 3:
                self.traffic_signals.append((event.xdata, event.ydata))
                self.map_num += 1
                for i in range(self.last_mapped + 1, len(self.starts)):
                    self.starts_to_traffic_signal.append(self.map_num)
                    print(f"Mapped {self.starts[i]} to {self.map_num}")
                self.last_mapped = len(self.starts) - 1
                self.ts_mapping.append(self.signal_num)
                print(f"{self.traffic_signals[-1]} --> {self.ts_mapping[-1]}")
                self.render_signals()
                print(self.traffic_signals)

        def onpress(self, event):
            print(event.key)
            sys.stdout.flush()
            if event.key == "1":
                self.starts = self.starts[:-1]
                plt.clf()
                self.render()
                self.render_pts()

            if event.key == "o":
                self.signal_num += 1
                self.signal_num %= 2
                print(f"Signal Flipped to {self.signal_num}")

            if event.key == "2":
                all_paths = maphelper.collect_paths(
                    map_name, self.starts, (midx, midy), width, height
                )
                for key in range(len(all_paths)):
                    paths = all_paths[key]
                    for pathi, path in enumerate(paths):
                        plt.plot(
                            [p[0] for p in path],
                            [p[1] for p in path],
                            f"C{pathi}",
                        )
                plt.draw()

            if event.key == "t":
                all_paths = maphelper.collect_paths(
                    map_name, self.starts, (midx, midy), width, height
                )
                outname = f"{map_name}_{midx}_{midy}.json"

                print(self.ts_mapping, self.starts_to_traffic_signal)

                print("saving", outname)
                info = {
                    "map_name": map_name,
                    "center": (midx, midy),
                    "width": width,
                    "height": height,
                    "all_paths": all_paths,
                    "starts": self.starts,
                    "road_img": road_img.tolist(),
                    "dx": dx.tolist(),
                    "bx": bx.tolist(),
                    "nx": nx,
                    "ny": ny,
                    "signal_locations": self.traffic_signals,
                    "mapping": self.ts_mapping,
                    "starts_to_signal": self.starts_to_traffic_signal,
                }
                with open(outname, "w") as writer:
                    json.dump(info, writer)
                plt.title(f"saved to {outname}!")
                plt.draw()

        def render_pts(self):
            plt.plot(
                [p[0] for p in self.starts],
                [p[1] for p in self.starts],
                "b.",
                markersize=15,
            )
            plt.draw()

        def render_signals(self):
            plt.plot(
                [p[0] for p in self.traffic_signals],
                [p[1] for p in self.traffic_signals],
                "g.",
                markersize=15,
            )
            plt.draw()

        def render(self):
            plt.plot(
                maphelper.trees[map_name]["tree"].data[:, 0],
                maphelper.trees[map_name]["tree"].data[:, 1],
                "k.",
                alpha=0.5,
            )
            plt.xlim((midx - width / 2, midx + width / 2))
            plt.ylim((midy - height / 2, midy + height / 2))
            ax = plt.gca()
            ax.set_aspect("equal")
            plt.draw()

    gui = GUI()
    gui.render()
    plt.show()


def fix_json_maps(glob_path="./*.json"):
    fs = glob(glob_path)
    for fi, f in enumerate(fs):
        print(f"Fixing {f}...")
        with open(f, "r") as reader:
            data = json.load(reader)

        # Some splines have repeated points. Clean those up
        for k, paths in data["all_paths"].items():
            new_paths = []
            for path in paths:
                path = np.array(path)
                new_paths.append(
                    [path[0].tolist()]
                    + path[1:][
                        (1 - (path[1:] == path[:-1]).all(-1)).astype(np.bool)
                    ].tolist()
                )
                if path.shape[0] != len(new_paths[-1]):
                    print(
                        f"[Point Cleanup] Before: {path.shape[0]} |"
                        f" After {len(new_paths[-1])}"
                    )
            data["all_paths"][k] = new_paths

        # Some splines merge into others. This causes issues in
        # the downstream map preprocessing code. We need to fuse
        # these
        for starti, (key, paths) in enumerate(data["all_paths"].items()):
            new_paths = []
            for j, path in enumerate(paths):
                path = np.array(path)[:, :2]
                complete_path = path.tolist()
                end = path[-1]
                done = False
                for i, p in enumerate(paths):
                    p = np.array(p)[:, :2]
                    if i == j:
                        continue
                    idxs = (end == p).all(axis=-1).nonzero()[0]
                    for idx in idxs:
                        if idx == p.shape[0] - 1:
                            continue
                        complete_path += p[(idx + 1) :].tolist()
                        print(
                            f"[Spline Fusion] Before: {path.shape[0]} "
                            f"| After: {len(complete_path)}"
                        )
                        done = True
                    if done:
                        break
                new_paths.append(complete_path)
            data["all_paths"][key] = new_paths

        with open(f, "w") as writer:
            json.dump(data, writer)


def preprocess_maps(dataroot, glob_path="./*.json"):
    fs = glob(glob_path)
    for fi, f in enumerate(fs):
        with open(f, "r") as reader:
            data = json.load(reader)
        nusc_map = NuScenesMap(dataroot=dataroot, map_name=data["map_name"])
        dataset = dict()
        center, h, w = data["center"], data["height"], data["width"]
        patch = [
            center[0] - w / 2,
            center[1] - h / 2,
            center[0] + w / 2,
            center[1] + h / 2,
        ]
        dataset["patch"] = patch
        dataset["center"] = np.array([center])
        dataset["height"] = h
        dataset["width"] = w
        dataset["map_name"] = data["map_name"]
        dataset["dx"] = np.array(data["dx"])
        dataset["bx"] = np.array(data["bx"])
        dataset["road_img"] = np.array(data["road_img"])

        # Needed for lidar sensors
        pt1, pt2 = nuscenes_map_to_line_representation(nusc_map, patch, False)
        dataset["edges"] = (pt1, pt2)

        drivable_area, xs, ys = get_drivable_area_matrix(data, patch, res=500)
        dataset["plotting_utils"] = (
            drivable_area.numpy().flatten(),
            xs.numpy().flatten(),
            ys.numpy().flatten(),
            [
                (0.5, 0.5, 0.5) if row else (1, 1, 1)
                for row in drivable_area.numpy().flatten()
            ],
        )

        dataset["splines"] = dict()
        signal_loc = torch.as_tensor(data["signal_locations"])
        signal_color = data["mapping"]
        dataset["signal_locations"] = signal_loc
        dataset["color_mapping"] = signal_color
        dataset["starts_to_signal"] = data["starts_to_signal"]
        dataset["signal_loc"] = []
        dataset["signal_color"] = []
        for starti, (key, paths) in enumerate(data["all_paths"].items()):
            idx = data["starts_to_signal"][starti]
            loc = signal_loc[idx]
            col = signal_color[idx]
            dataset["signal_loc"].append(loc)
            dataset["signal_color"].append(col)

            dataset["splines"][starti] = dict()
            for pathi, path in enumerate(paths):
                path = np.array(path)
                if path.shape[0] < 75:
                    print(
                        "Skipping spline as it contains very few control points"
                    )
                    continue
                dataset["splines"][starti][pathi] = []
                for i in range(0, 50, 10):
                    cps = path[
                        np.linspace(i, path.shape[0] - 15, 12, dtype=np.int),
                        :2,
                    ]

                    diff = cps[0] - cps[1]
                    theta = np.arctan2(diff[1], diff[0])
                    start_orientation = (
                        angle_normalize(torch.as_tensor(math.pi + theta))
                        .float()
                        .reshape(1, 1)
                    )
                    if i == 0:
                        extra_pt1 = np.array(
                            [
                                [
                                    cps[0, 0] + np.cos(theta) * 30.0,
                                    cps[0, 1] + np.sin(theta) * 30.0,
                                ]
                            ]
                        )
                    else:
                        extra_pt1 = path[0:1, :2]

                    diff = cps[-1] - cps[-2]
                    theta = np.arctan2(diff[1], diff[0])
                    dest_orientation = (
                        angle_normalize(torch.as_tensor(theta))
                        .float()
                        .reshape(1, 1)
                    )
                    extra_pt2 = np.array(
                        [
                            [
                                cps[-1, 0] + np.cos(theta) * 30.0,
                                cps[-1, 1] + np.sin(theta) * 30.0,
                            ]
                        ]
                    )

                    cps = torch.cat(
                        [
                            torch.from_numpy(cps),
                            torch.from_numpy(extra_pt2),
                            torch.from_numpy(extra_pt1),
                        ]
                    )[None, :, :].float()

                    start_position = cps[:, 0, :]
                    destination = cps[:, -3, :]

                    dataset["splines"][starti][pathi].append(
                        (
                            start_position,
                            destination,
                            start_orientation,
                            dest_orientation,
                            cps,
                        )
                    )
        outname = f"env{data['map_name']}_{data['center'][0]}_{data['center'][1]}.pth"
        print("saving", outname)
        torch.save(dataset, outname)


def viz_nuscenes_world(
    glob_path="./*.pth",
    as_pdf: bool = True,
    montage: bool = False,
    save_path: Optional[str] = "all_maps.pdf",
):
    # FIXME: Montage doesn't work as of now
    fs = glob(glob_path)
    if montage:
        nrow = int(np.ceil(np.sqrt(len(fs))))
        ncol = int(np.ceil(len(fs) / nrow))
        fig, axs = plt.subplots(nrow, ncol, figsize=(10, 10))
        i = 0
        j = 0
    for fi, f in enumerate(fs):
        data = torch.load(f)
        world = NuscenesWorld(f)
        if montage:
            world.render(
                fig=fig, ax=axs[i, j] if not ncol == nrow == 1 else axs
            )
            j += 1
            if j % ncol == 0:
                j = 0
                i += 1
        else:
            world.render()
        world.ax.set_aspect("equal")
        world.ax.grid(False)
        world.ax.set_xticklabels([])
        world.ax.set_yticklabels([])
        plt.tight_layout()

        if not montage:
            if as_pdf:
                world.fig.set_rasterized(True)
                outname = f"env{data['map_name']}_{data['center'][0][0]}_{data['center'][0][1]}.pdf"
            else:
                outname = f"env{data['map_name']}_{data['center'][0][0]}_{data['center'][0][1]}.png"
            print("saving", outname)
            plt.savefig(outname, bbox_inches="tight")
            plt.close(world.fig)
    if montage:
        fig.set_rasterized(True)
        fig.savefig(save_path, bbox_inches="tight")


def viz_env(glob_path="./*.json"):
    fs = glob(glob_path)
    for fi, f in enumerate(fs):
        with open(f, "r") as reader:
            data = json.load(reader)

        fig = plt.figure()
        gs = mpl.gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

        # plot the drivable area
        xs = np.array(
            [
                np.linspace(
                    data["center"][0] - data["width"] / 2 * 1.1,
                    data["center"][0] + data["width"] / 2 * 1.1,
                    100,
                )
                for _ in range(100)
            ]
        ).flatten()
        ys = np.array(
            [
                np.linspace(
                    data["center"][1] - data["height"] / 2 * 1.1,
                    data["center"][1] + data["height"] / 2 * 1.1,
                    100,
                )
                for _ in range(100)
            ]
        ).T.flatten()
        # index into drivable_area
        drivable_area = np.array(data["road_img"])
        ixes = (
            np.array([xs, ys]).T
            - np.array([data["center"]])
            - np.array(data["bx"])[:2]
            + np.array(data["dx"])[:2] / 2.0
        ) / np.array(data["dx"])[:2]
        ixes = ixes.astype(int)
        within = np.logical_and(0 <= ixes[:, 0], ixes[:, 0] < data["nx"])
        within = np.logical_and(within, 0 <= ixes[:, 1])
        within = np.logical_and(within, ixes[:, 1] < data["ny"])
        drivable = np.zeros(len(ixes))
        drivable[within] = drivable_area[ixes[within, 0], ixes[within, 1]]
        c = [(0, 1, 0) if row else (1, 0, 0) for row in drivable]
        plt.scatter(xs, ys, alpha=0.05, c=c)

        # plot each path
        for starti, (key, paths) in enumerate(data["all_paths"].items()):
            # each path emanating from this start position
            for path in paths:
                plt.plot(
                    [p[0] for p in path],
                    [p[1] for p in path],
                    c=f"C{starti}",
                    alpha=0.5,
                )

        # plot each start position (N x 2)
        starts = np.array(data["starts"])
        plt.plot(
            starts[:, 0],
            starts[:, 1],
            ".",
            markersize=10,
            label="Start Positions",
        )

        for loc, col in zip(data["signal_locations"], data["mapping"]):
            plt.plot(
                [loc[0]], [loc[1]], "g." if col == 1 else "r.", markersize=15
            )

        # make the window slightly larger than the actual boundaries for viz
        fac = 1.1
        plt.xlim(
            (
                data["center"][0] - data["width"] * fac / 2,
                data["center"][0] + data["width"] * fac / 2,
            )
        )
        plt.ylim(
            (
                data["center"][1] - data["height"] * fac / 2,
                data["center"][1] + data["height"] * fac / 2,
            )
        )
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.tight_layout()

        outname = f"env{data['map_name']}_{data['center'][0]}_{data['center'][1]}.jpg"
        print("saving", outname)
        plt.savefig(outname, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    Fire(
        {
            "env_create": env_create,
            "find_center": find_center,
            "viz_env": viz_env,
            "viz_nuscenes_world": viz_nuscenes_world,
            "preprocess_maps": preprocess_maps,
            "fix_json_maps": fix_json_maps,
        }
    )
