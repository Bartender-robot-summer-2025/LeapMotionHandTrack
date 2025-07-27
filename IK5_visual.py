import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from scipy.optimize import minimize
import meshcat
import os, socket, json, threading
import meshcat.geometry as g
import time

# Import LEAP Hand control modules
try:
    from leap_hand_utils.dynamixel_client import *
    import leap_hand_utils.leap_hand_utils as lhu
    LEAP_AVAILABLE = True
except ImportError:
    print("Warning: LEAP Hand modules not available. Running in visualization-only mode.")
    LEAP_AVAILABLE = False

# === LEAP Hand Control Class ===
class LeapHandController:
    def __init__(self):
        if not LEAP_AVAILABLE:
            self.available = False
            return
            
        self.available = True
        # LEAP Hand parameters
        self.kP = 300
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350  # Use 550 for full motors
        self.motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        
        try:
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                try:
                    self.dxl_client = DynamixelClient(self.motors, 'COM13', 4000000)
                    self.dxl_client.connect()
                except Exception as e:
                    print(f"Failed to connect to LEAP Hand: {e}")
                    self.available = False
                    return
        
        # Initialize LEAP Hand
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
        
        # Set initial position (open hand)
        self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        print("LEAP Hand initialized successfully!")
    
    def set_allegro_pose(self, pose):
        """Set LEAP Hand using Allegro-compatible joint angles"""
        if not self.available:
            return
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        
        # Fix joint reversal issues: swap first and second joints for each finger
        fixed_pose = np.array(pose)
        # Index finger (joints 0-3): swap 0 and 1
        fixed_pose[0], fixed_pose[1] = fixed_pose[1], fixed_pose[0]
        # Ring finger (joints 12-15): swap 12 and 13
        fixed_pose[12], fixed_pose[13] = fixed_pose[13], fixed_pose[12]
        
        fixed_pose[8], fixed_pose[9] = fixed_pose[9], fixed_pose[8]
        
        # Reorder joint array: move thumb to last 4 indices
        # Original: index(0-3), thumb(4-7), middle(8-11), ring(12-15)
        # New: index(0-3), middle(4-7), ring(8-11), thumb(12-15)
        reordered_pose = np.zeros(16)
        reordered_pose[0:4] = fixed_pose[0:4]    # index stays at 0-3
        reordered_pose[4:8] = fixed_pose[8:12]   # middle moves from 8-11 to 4-7
        reordered_pose[8:12] = fixed_pose[12:16] # ring moves from 12-15 to 8-11
        reordered_pose[12:16] = fixed_pose[4:8]  # thumb moves from 4-7 to 12-15
        
        self.curr_pos = reordered_pose
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    
    def read_pos(self):
        """Read current joint positions"""
        if not self.available:
            return np.zeros(16)
        return self.dxl_client.read_pos()

# === Paths and Models (LeapHand) ===
model_path = os.path.abspath("/home/atrc234/PinnoMeshHand/leap_hand/stl")
urdf_path = os.path.join(os.path.dirname(model_path), "robot.urdf")

robot = RobotWrapper.BuildFromURDF(urdf_path, [model_path])
model, data = robot.model, robot.data

# === Initialize Meshcat Visualizer ===
viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# === Joint Limits and Frame Definitions for LeapHand ===

leap_dof_lower = np.array([-0.5, -1.047, -0.506, -0.366,
                        -0.47, -0.349, -1.20, -1.34,
                        -0.5, -1.047, -0.506, -0.366,
                        -0.5, -1.047, -0.506, -0.366])
leap_dof_upper = np.array([0.5, 1.047, 1.885, 2.042,
                        2.443, 2.094, 1.90, 1.88,
                        0.5, 1.047, 1.885, 2.042,
                        0.5, 1.047, 1.885, 2.042])
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
    # Initialize LEAP Hand controller
    leap_controller = LeapHandController()
    
    # No ROS initialization
    receiver = UDPReceiver()
    threading.Thread(target=receiver.listen, daemon=True).start()
    delta_q_max = 0.2

    print("Start multi-finger control (LEAP Hand + visualization)")
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
                
                # Add z-offset for thumb to increase its height
                if finger['name'] == 'thumb':
                    target[2] += 0.05  # Add 5cm to thumb z position

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

            # Send IK solution to LEAP Hand hardware
            if leap_controller.available:
                leap_controller.set_allegro_pose(q_final)
                # Optional: read and display actual joint positions
                actual_pos = leap_controller.read_pos()
                print(f"Actual LEAP positions: {actual_pos}")
            
            viz.display(q_final)
            time.sleep(0.01)  # 10Hz instead of 33Hz

    except KeyboardInterrupt:
        print("Manually interrupted, exiting.")
        # Set hand to open position before exiting
        if leap_controller.available:
            leap_controller.set_allegro_pose(np.zeros(16))

if __name__ == "__main__":
    control_loop() 