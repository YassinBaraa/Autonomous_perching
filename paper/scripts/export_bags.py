#!/usr/bin/env python3
"""Export the topics the paper needs from ROS1 bags to CSV (runs inside the uav_ros_stack container).

usage: python3 export_bags.py OUT_DIR BAG [BAG ...]
Writes OUT_DIR/<bag name>/<topic>.csv; column t = bag record time [s, ROS/wall clock]."""
import csv
import math
import os
import sys

import rosbag

NS = "/red"


def pt(m):
    return [m.header.stamp.to_sec(), m.point.x, m.point.y, m.point.z]


def odom(m):
    p, q, v, w = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear, m.twist.twist.angular
    return [m.header.stamp.to_sec(), p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z]


def imu(m):
    q, w, a = m.orientation, m.angular_velocity, m.linear_acceleration
    return [m.header.stamp.to_sec(), q.x, q.y, q.z, q.w, w.x, w.y, w.z, a.x, a.y, a.z]


def att(m):
    q, r = m.orientation, m.body_rate
    return [m.header.stamp.to_sec(), q.x, q.y, q.z, q.w, r.x, r.y, r.z, m.thrust]


def esc(m):
    row = [m.header.stamp.to_sec()]
    for e in list(m.esc_telemetry)[:4]:
        row += [e.rpm, e.voltage, e.current, e.temperature]
    return row


PT = ["stamp", "x", "y", "z"]
TOPICS = {
    "ibvs/state": (["data"], lambda m: [m.data]),
    "ibvs/approach": (["data"], lambda m: [m.data]),
    "ibvs/target_point": (PT, pt),
    "ibvs/target_point_virtual": (PT, pt),
    "ibvs/error": (PT, pt),
    "ibvs/closeness": (PT, pt),
    "ibvs/contact": (PT, pt),
    "ibvs/height": (PT, pt),
    "ibvs/size_height": (PT, pt),
    "ibvs/thrust": (PT, pt),
    "ibvs/control_angles": (PT, pt),
    "ibvs/pid_roll": (PT, pt),
    "ibvs/pid_pitch": (PT, pt),
    "ibvs/pid_z": (PT, pt),
    "ibvs/gripper": (PT, pt),
    "ibvs/virtual_camera/tilt": (PT, pt),
    "vrpn_client/estimated_odometry": (["stamp", "px", "py", "pz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz"], odom),
    "mavros/local_position/odom": (["stamp", "px", "py", "pz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz"], odom),
    "mavros/imu/data": (["stamp", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az"], imu),
    "mavros/battery": (["voltage", "current", "percentage"], lambda m: [m.voltage, m.current, m.percentage]),
    "mavros/state": (["armed", "mode"], lambda m: [int(m.armed), m.mode]),
    "mavros/setpoint_raw/attitude": (["stamp", "qx", "qy", "qz", "qw", "rx", "ry", "rz", "thrust"], att),
    "mavros/esc_telemetry": (["stamp"] + [f"{k}{i}" for i in range(4) for k in ("rpm", "v", "a", "temp")], esc),
}


def main():
    out, bags = sys.argv[1], sys.argv[2:]
    for path in bags:
        name = os.path.splitext(os.path.basename(path))[0]
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        files, writers = {}, {}
        wanted = {NS + "/" + k: k for k in TOPICS}
        n = 0
        with rosbag.Bag(path) as bag:
            for topic, msg, t in bag.read_messages(topics=list(wanted)):
                key = wanted[topic]
                if key not in writers:
                    f = open(os.path.join(d, key.replace("/", "__") + ".csv"), "w", newline="")
                    files[key], writers[key] = f, csv.writer(f)
                    writers[key].writerow(["t"] + TOPICS[key][0])
                try:
                    row = TOPICS[key][1](msg)
                except Exception:
                    continue
                writers[key].writerow([f"{t.to_sec():.6f}"] + [f"{v:.6f}" if isinstance(v, float) and math.isfinite(v) else v for v in row])
                n += 1
        for f in files.values():
            f.close()
        print(f"{name}: {n} messages, topics {sorted(files)}", flush=True)


if __name__ == "__main__":
    main()
