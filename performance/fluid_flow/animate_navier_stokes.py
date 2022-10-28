#!/usr/bin/env python
# Al Danial, October 2022
# https://github.com/AlDanial/matlab_with_Python/
# This code is covered by the MIT open source license.

# Create animation or individual PNG images of fluid flow previously
# computed with py_Main.py or Main.m.

import sys
import os.path
import argparse
from glob import glob
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from navier_stokes_functions import VELOCITY_OUT_DIR, PREFIX

def parse_args():  # {{{
    parser = argparse.ArgumentParser(
        description=f"""Read a directory of files with names
   {PREFIX}vx_*.mat,
   {PREFIX}vy_*.mat
 as produced by Main.m in
 https://github.com/JamieMJohns/Navier-stokes-2D-numerical-solve-incompressible-flow-with-custom-scenarios-MATLAB-.git
 These .mat files are typically in the run subdirectory
   ./temporary_NS_velocity"""
    )

    parser.add_argument(
        "mat_dir",
        metavar="MAT_DIR",
        type=str,
        nargs="*",
        default=[VELOCITY_OUT_DIR],
        help=f"Directory of {PREFIX}*.mat files. " f"[default {VELOCITY_OUT_DIR}]",
    )

    parser.add_argument(
        "-a",
        "--archive",
        dest="merged_archive",
        action="store",
        type=str,
        default="merged_frames.npy",
        help="Name of the merged .npy file " '[default "merged_frames.npy"].',
    )

    parser.add_argument(
        "--caption",
        dest="caption",
        action="store",
        type=str,
        default=None,
        help="Caption below figure.",
    )

    parser.add_argument(
        "-i",
        "--incr",
        dest="incr",
        action="store",
        type=int,
        default=1,
        help="Frame increment for playback and for writing "
        "the merged npy file [default 1].",
    )

    parser.add_argument(
        "-f",
        "--frame-time",
        dest="frame_time_file",
        action="store",
        type=str,
        default="",
        help="Name of frame time .npy file.",
    )

    parser.add_argument(
        "-n",
        "--npy",
        dest="npy_file",
        action="store",
        type=str,
        default="",
        help="Name of existing .npy file to animate.  "
        "Also supply --frame-time FT.npy",
    )

    parser.add_argument(
        "-m",
        "--matlab",
        dest="matlab",
        action="store_true",
        default=False,
        help="Load velocities from MATLAB .mat files. " "matlab .mat files.",
    )

    parser.add_argument(
        "-p",
        "--python",
        dest="python",
        action="store_true",
        default=False,
        help="Load velocities from Python .npy files.",
    )

    parser.add_argument(
        "--save-only",
        dest="save_only",
        action="store_true",
        default=False,
        help="Write the merged npy file (for use with " "--npy) then exit.",
    )

    parser.add_argument(
        "--png-dir",
        dest="png_dir",
        metavar="PNGDIR",
        action="store",
        type=str,
        default="PNG",
        help="Save each frame to a PNG file in the given " "subdirectory.",
    )

    parser.add_argument(
        "-s",
        "--start-frame",
        dest="start_frame",
        action="store",
        type=int,
        default=0,
        help="Start frame index [default 0].",
    )

    parser.add_argument(
        "--start-offset",
        metavar="EL_IND",
        nargs=2,
        action="store",
        default=(0, 0),
        help="Only used with --png-dir.  Two values:  number of "
        "seconds and start index to add to elapsed time in the PNG "
        "title and to add to the frame index in the filename ["
        "default=(0,0)].",
    )

    parser.add_argument(
        "-t",
        "--title",
        dest="title",
        action="store",
        type=str,
        default=None,
        help='Plot title prefix.  Default is "Python" '
        'if running with --python, "MATLAB" if running '
        "with --matlab, blank with --npy.",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if not args.python and not args.matlab and not args.npy_file:
        print("Specify -p/--python, -m/--matlab, or -n/--npy to define the ")
        print("type of velocity files to load: *.npy, *.mat, or merged .npy")
        sys.exit(0)

    ext = None
    title = args.title
    if args.python:
        # individual .npy, not merged .npy
        ext = "npy"
        if not title:
            title = "Python"
    elif args.matlab:
        ext = "mat"
        if not title:
            title = "MATLAB"

    if (args.png_dir is not None) and (not os.path.exists(args.png_dir)):
        os.mkdir(args.png_dir)

    return args, ext, title

# }}}
def combine_mat_frames(x_mats, y_mats, incr=1):  # {{{
    vel_list = []
    n_frames = 0
    for x_file, y_file in zip(x_mats, y_mats):
        print(x_file)
        xm = loadmat(x_file, squeeze_me=True)
        ym = loadmat(y_file, squeeze_me=True)
        xv = xm["var"][:, :, ::incr]
        yv = ym["var"][:, :, ::incr]
        vel_list.append(np.sqrt(xv**2 + yv**2))
        nX, nY, nFr = xv.shape
        n_frames += nFr

    all_vel = np.zeros((nX, nY, n_frames), dtype=np.float32)
    i = 0
    for vel in vel_list:
        nX, nY, nFr = vel.shape
        all_vel[:, :, i : i + nFr] = vel
        i += nFr
    return all_vel

# }}}
def combine_npy_frames(x_mats, y_mats, incr=1):  # {{{
    vel_list = []
    n_frames = 0
    if not x_mats:
        print("No npy files were loaded")
        exit(1)
    for x_file, y_file in zip(x_mats, y_mats):
        print(x_file)
        xv = np.load(x_file)
        nX, nY, nFr = xv.shape
        if incr > nFr:
            print(f"--incr {incr} is too large ({x_file} only has {nFr} frames)")
            sys.exit(0)
        xv = xv[:, :, ::incr]
        nFr = xv.shape[2]  # number of frames in the reduced set
        yv = np.load(y_file)[:, :, ::incr]
        vel_list.append(np.sqrt(xv**2 + yv**2))
        n_frames += nFr

    all_vel = np.zeros((nX, nY, n_frames), dtype=np.float32)
    i = 0
    for vel in vel_list:
        nX, nY, nFr = vel.shape
        all_vel[:, :, i : i + nFr] = vel
        i += nFr
    return all_vel

# }}}
def animate_flow(  # {{{
    vel,
    title,
    frame_time=None,
    caption=None,
    start_frame=0,
    incr=1,
    png_dir=None,
    start_offset=(0, 0),
):
    start_elapsed_time = float(start_offset[0])
    start_i_frame = int(start_offset[1])
    fig, ax = plt.subplots()
    im = plt.contourf(vel[:, :, 0], origin="upper")
    ax.set_aspect("equal")
    n_frames = vel.shape[-1]
    i_frame = start_frame

    def update(*args):  # {{{2
        nonlocal i_frame, im
        for tp in im.collections:
            try:
                tp.remove()
            except ValueError:
                sys.exit(0)
        if i_frame >= n_frames:
            if png_dir is not None:
                return None  # brute force exit w/error, only do PNG's once
            i_frame = start_frame
        im = plt.contourf(
            vel[:, :, i_frame],
            vmin=0,
            vmax=max_vel * 0.8,
            levels=20,
            cmap="cividis",
            origin="upper",
        )
        if frame_time is not None:
            elapsed, sim = frame_time[i_frame, :]
            ax.set_title(
                f"{title}            Elapsed: "
                f"{elapsed + start_elapsed_time:7.3f} s      "
                f"Sim: {sim:7.3f} s",
                fontsize=6,
                loc="right",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        if caption is not None:
            fig.text(0.5, 0.005, caption)
        plt.tight_layout()
        if png_dir is not None:
            png_file = f"{png_dir}/frame_{i_frame + start_i_frame:05d}.png"
            fig.savefig(
                png_file,
                dpi=250,
                bbox_inches="tight",
                pad_inches=0.1,
            )
        i_frame += incr
        print(f"i_frame={i_frame} {png_file}")
        return (im,)

    # 2}}}
    max_vel = vel.max()
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=10
    )  # , blit=True)
    plt.show()


# }}}
def main():  # {{{

    args, ext, title = parse_args()
    tmp_NS_dir = args.mat_dir[0]
    frame_time = None
    if ext in {"npy", "mat"}:
        x_vel_files = sorted(glob(f"{tmp_NS_dir}/{PREFIX}vx_*.{ext}"))
        y_vel_files = sorted(glob(f"{tmp_NS_dir}/{PREFIX}vy_*.{ext}"))
        frame_time_file = f"{tmp_NS_dir}/{PREFIX}ft.{ext}"
    if args.npy_file:
        frame_time_file = args.frame_time_file
        vel = np.load(args.npy_file)
    elif args.python:
        vel = combine_npy_frames(x_vel_files, y_vel_files, incr=args.incr)
        np.save(args.merged_archive, vel)
        if args.save_only:
            sys.exit(0)
        args.incr = 1  # reset since *_npys already subsetted by incr
    else:
        vel = combine_mat_frames(x_vel_files, y_vel_files, incr=args.incr)
        np.save(args.merged_archive, vel)
        if args.save_only:
            sys.exit(0)
        args.incr = 1  # reset since *_mats already subsetted by incr
    nX, nY, nFr = vel.shape
    print(
        f"vel  max= {vel.max():.3f}  MB={vel.nbytes/1024**2:.3f}  "
        f"{nFr} frames of {nX} x {nY}"
    )

    if os.path.exists(frame_time_file):
        if frame_time_file.endswith(".npy"):
            frame_time = np.load(frame_time_file)
        else:
            mat_data = loadmat(frame_time_file, squeeze_me=True)
            frame_time = mat_data["var"]
    animate_flow(
        vel,
        title,
        frame_time=frame_time,
        caption=args.caption,
        start_frame=args.start_frame,
        incr=args.incr,
        png_dir=args.png_dir,
        start_offset=args.start_offset,
    )

# }}}

if __name__ == "__main__":
    main()
