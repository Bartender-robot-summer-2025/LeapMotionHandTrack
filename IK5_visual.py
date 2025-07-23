import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from scipy.optimize import minimize
import meshcat
import os, socket, json, threading
import meshcat.geometry as g
# import rospy  # Removed ROS
# from allegro import Allegro  # Not used

# === Paths and Models (LeapHand) ===
model_path = os.path.abspath("/home/atrc234/PinnoMeshHand/leap_hand/stl")
urdf_path = os.path.join(os.path.dirname(model_path), "robot.urdf")

robot = RobotWrapper.BuildFromURDF(urdf_path, [model_path])
model, data = robot.model, robot.data

# === Initialize Meshcat Visualizer ===
viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()
'''
# === Joint Limits and Frame Definitions for LeapHand ===
leap_dof_lower = np.array([
    -0.314, -1.047, -0.506, -0.366,   # index: 1,0,2,3
    -0.349, -0.47, -1.20, -1.34,       # thumb: 12,13,14,15
    -0.314, -1.047, -0.506, -0.366,   # middle: 5,4,6,7
    -0.314, -1.047, -0.506, -0.366   # ring: 9,8,10,11
    
])
leap_dof_upper = np.array([
    2.23, 1.047, 1.885, 2.042,        # index
    2.094, 2.443, 1.90, 1.88,          # thumb
    2.23, 1.047, 1.885, 2.042,        # middle
    2.23, 1.047, 1.885, 2.042       # ring
])
'''
'''
leap_dof_lower = np.array([-1.047, -0.314, -0.506, -0.366,
                        -1.047, -0.314, -0.506, -0.366,
                        -1.047, -0.314, -0.506, -0.366,
                        -0.349, -0.47, -1.20, -1.34])
leap_dof_upper = np.array([1.047, 2.23, 1.885, 2.042,
                        1.047, 2.23, 1.885, 2.042,
                        1.047, 2.23, 1.885, 2.042,
                        2.094, 2.443, 1.90, 1.88])
'''
leap_dof_lower = np.array([-1.047, -0.314, -0.506, -0.366,
                        -0.349, -0.47, -1.20, -1.34,
                        -1.047, -0.314, -0.506, -0.366,
                        -1.047, -0.314, -0.506, -0.366])
leap_dof_upper = np.array([1.047, 2.23, 1.885, 2.042,
                        2.094, 2.443, 1.90, 1.88,
                        1.047, 2.23, 1.885, 2.042,
                        1.047, 2.23, 1.885, 2.042])
# LeapHand finger mapping and frame names, in UDP order: thumb, index, middle, ring
fingers = [
    {"name": "thumb",  "frame": "thumb_fingertip_tip", "q_range": slice(4, 8)},
    {"name": "index",  "frame": "fingertip_tip",      "q_range": slice(0, 4)},
    {"name": "middle", "frame": "fingertip_2_tip",    "q_range": slice(8, 12)},
    {"name": "ring",   "frame": "fingertip_3_tip",    "q_range": slice(12, 16)},
]

# Initialize per-finger joint limits and frame IDs
for finger in fingers:
    q_slice = finger["q_range"]
    finger["lower"] = leap_dof_lower[q_slice]
    finger["upper"] = leap_dof_upper[q_slice]
    finger["frame_id"] = model.getFrameId(finger["frame"])
    finger["q_prev"] = np.clip(np.zeros(4), finger["lower"], finger["upper"])

# === UDP Receiver ===
class UDPReceiver:
    def __init__(self, ip="0.0.0.0", port=12345):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)
        self.target_positions = [np.array([0.05, 0, 0])] * 4

    def listen(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                print("Received UDP:", data)
                arr = json.loads(data.decode())
                if isinstance(arr, list) and len(arr) >= 4:
                    self.target_positions = [np.asarray(p, dtype=float) for p in arr[:4]]
            except BlockingIOError:
                pass
            except Exception as e:
                print("UDP error:", e)

# === Visualization Control Loop ===
def control_loop():
    # No ROS initialization
    receiver = UDPReceiver()
    threading.Thread(target=receiver.listen, daemon=True).start()
    delta_q_max = 0.2

    print("Start multi-finger control (visualization only, LeapHand)")
    try:
        while True:
            q_final = np.zeros(model.nq)

            # Use only the first 4 positions from UDP (thumb, index, middle, ring)
            udp_finger_targets = receiver.target_positions[:4]
            for i, finger in enumerate(fingers):
                q_slice = finger["q_range"]
                lower, upper = finger["lower"], finger["upper"]
                frame_id = finger["frame_id"]
                q_prev = finger["q_prev"]
                ori = udp_finger_targets[i]
                print(ori)
                target = np.array([-ori[1], -ori[0], -ori[2]*1.7], dtype=float) / 1000.0 * 2

                def cost_fn(q_local):
                    q_all = np.zeros(model.nq)
                    q_all[q_slice] = q_local
                    pin.forwardKinematics(model, data, q_all)
                    pin.updateFramePlacement(model, data, frame_id)
                    pos = data.oMf[frame_id].translation
                    return np.linalg.norm(pos - target)

                def step_constraint(q_local):
                    return delta_q_max**2 - np.linalg.norm(q_local - q_prev)**2

                res = minimize(
                    cost_fn,
                    q_prev,
                    method="SLSQP",
                    bounds=list(zip(lower, upper)),
                    constraints=[{"type": "ineq", "fun": step_constraint}],
                    options={"ftol": 1e-6, "maxiter": 80}
                )
                q_sol = np.clip(res.x, lower, upper)
                q_final[q_slice] = q_sol
                finger["q_prev"] = q_sol

                viz.viewer[f"target_{finger['name']}"].set_object(
                    meshcat.geometry.Sphere(0.01),
                    meshcat.geometry.MeshLambertMaterial(color=0xff0000)
                )
                viz.viewer[f"target_{finger['name']}"].set_transform(
                    pin.SE3(np.eye(3), target).homogeneous
                )

                # Add yellow sphere at the current fingertip position
                pin.forwardKinematics(model, data, q_final)
                pin.updateFramePlacement(model, data, frame_id)
                fingertip_pos = data.oMf[frame_id].translation
                fingertip_rot = data.oMf[frame_id].rotation
                viz.viewer[f"fingertip_{finger['name']}_actual"].set_object(
                    meshcat.geometry.Sphere(0.01),
                    meshcat.geometry.MeshLambertMaterial(color=0xffff00)
                )
                viz.viewer[f"fingertip_{finger['name']}_actual"].set_transform(
                    pin.SE3(np.eye(3), fingertip_pos).homogeneous
                )

       
            viz.display(q_final)
            # Visualization only, no hardware or ROS

    except KeyboardInterrupt:
        print("Manually interrupted, exiting.")

if __name__ == "__main__":
    control_loop() 