import leap  # Import the Leap Motion SDK
import numpy as np  # For numerical calculations
import cv2  # For image display
import socket  # For UDP communication
import json  # For data serialization

# Define the Canvas class for hand skeleton visualization and UDP transmission
class Canvas:
    def __init__(self):
        # Initialize basic attributes
        self.name = "Leap Motion Skeleton Visualiser"  # Window title
        self.screen_size = [500, 700]  # Canvas size (height, width)
        self.hands_colour = (255, 255, 255)  # Hand color (white)
        self.font_colour = (0, 255, 44)  # Font color (green)
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)  # Black background image

        # Set UDP target
        self.udp_ip = "192.168.0.172"  # Receiver IP address
        self.udp_port = 12345  # Port number
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create UDP socket

    # Compute quaternion conjugate (used for inverse rotation)
    def quaternion_conjugate(self, q):
        return [q[0], -q[1], -q[2], -q[3]]

    # Quaternion multiplication (composite rotation)
    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ]

    # Rotate vector using quaternion
    def quaternion_rotate_vector(self, q, v):
        v_quat = [0.0] + list(v)
        qv = self.quaternion_multiply(q, v_quat)
        rotated_v = self.quaternion_multiply(qv, self.quaternion_conjugate(q))
        return rotated_v[1:]  # Return rotated 3D vector

    # Normalize quaternion
    def quaternion_normalize(self, q):
        norm = np.linalg.norm(q)
        return [q_i / norm for q_i in q]

    # Get projected position of joint on the canvas
    def get_joint_position(self, joint, hand_position, hand_orientation):
        if joint:
            joint_position = (joint.x, joint.y, joint.z)
            v = np.array(joint_position) - np.array(hand_position)
            q = self.quaternion_normalize(hand_orientation)
            v_rotated = self.quaternion_rotate_vector(q, v)
            x_screen = int(v_rotated[0] + (self.screen_size[1] / 2))
            y_screen = int(v_rotated[2] + (self.screen_size[0] / 2))
            return x_screen, y_screen, v_rotated
        else:
            return None

    # Compute 3D angle (with direction)
    def calculate_angle_3D(self, vector1, vector2, reference_axis=[0, 0, 1]):
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        ref_axis = np.array(reference_axis)
        dot_product = np.dot(v1, v2)
        cos_theta = np.clip(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        sign = np.sign(np.dot(np.cross(v1, v2), ref_axis))
        return np.degrees(angle_rad) * sign

    # Compute projected angle (with direction)
    def calculate_angle(self, vec1, vec2, reference_axis=[-1, 1, 0]):
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        dot_product = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        sign = np.dot(np.cross(vec1_norm, vec2_norm), reference_axis)
        if sign < 0:
            angle_rad = -angle_rad
        return np.degrees(angle_rad)

    # Send angle data via UDP
    def send_angles_udp(self, angles):
        try:
            data = json.dumps(angles)
            self.sock.sendto(data.encode(), (self.udp_ip, self.udp_port))
        except Exception as e:
            print(f"Error sending UDP data: {e}")

    def render_hands(self, event):
        """
        Render/update a complete hand skeleton and joint angle info on the canvas image.

        Parameters
        ----------
        event : leap.Frame
            The event object passed from each Leap Motion frame, containing all detected hands.
        """
        # ★ 1. Clear previous frame to avoid ghosting
        self.output_image[:, :] = 0            # Fill image with black
        text_offset = 20                       # Y offset for text display

        # ★ 2. Loop through all detected hands (usually 0–2)
        for i in range(len(event.hands)):
            hand = event.hands[i]

            # ---- (1) Get palm 3D position (in mm) ----
            hand_position = (
                hand.palm.position.x,
                hand.palm.position.y,
                hand.palm.position.z
            )

            # ---- (2) Convert Leap quaternion (x, y, z, w) → (w, -x, -y, -z) format for ROS/OpenCV
            quaternion = [
                hand.palm.orientation[3],      # w
                -hand.palm.orientation[0],     # -x
                -hand.palm.orientation[1],     # -y
                -hand.palm.orientation[2]      # -z
            ]

            fingers_angles = []                # Store joint angles of all 5 fingers
            fingers_pose = []

            # ★ 3. Loop through thumb (0) + four fingers (1~4)
            for index_digit in range(5):
                digit = hand.digits[index_digit]
                bone_vectors = []              # Each finger has 4 bone vectors
                finger_angles = []             # Joint angles for one finger

                # === 3.1 Special case: Thumb (anatomically different) ===
                if index_digit == 0:
                    bone = digit.bones[3]  # Distal bone
                    tip_joint = bone.next_joint  # Fingertip
                    tip_pos = self.get_joint_position(tip_joint, hand_position, quaternion)
                    fingers_pose.append(tip_pos[2])

                    i = 1  # Used to treat first thumb bone specially
                    for index_bone in range(4):
                        bone = digit.bones[index_bone]
                        # --- ① Get joint coordinates and local 3D vectors ---
                        start_data = self.get_joint_position(bone.prev_joint, hand_position, quaternion)
                        end_data   = self.get_joint_position(bone.next_joint, hand_position, quaternion)

                        if start_data and end_data:
                            start_xyz = np.array(start_data[2])
                            end_xyz   = np.array(end_data[2])

                            # --- ② Compute bone vector: first thumb segment fixed toward +Z ([0,0,-1] = -Z) ---
                            bone_vector = [0, 0, -1] if i == 1 else end_xyz - start_xyz
                            bone_vectors.append(bone_vector)

                            # --- ③ Draw bone: white line + endpoint circle ---
                            cv2.line(self.output_image,
                                     (start_data[0], start_data[1]),
                                     (end_data[0], end_data[1]),
                                     self.hands_colour, 2)
                            cv2.circle(self.output_image,
                                       (end_data[0], end_data[1]),
                                       3, self.hands_colour, -1)
                        i += 1

                    # --- ④ Compute thumb's 3 joint angles ---
                    for j in range(len(bone_vectors) - 1):
                        vec1 = bone_vectors[j]
                        vec2 = bone_vectors[j + 1]

                        # Project onto YZ plane → measures flexion/extension
                        vec1_yz = vec1.copy(); vec1_yz[0] = 0
                        vec2_yz = vec2.copy(); vec2_yz[0] = 0

                        if j == 0:
                            # MCP also measures abduction/adduction (XZ plane)
                            finger_angles.append(self.calculate_angle(vec1_yz, vec2_yz))
                        else:
                            # IP joints use true 3D angle
                            finger_angles.append(self.calculate_angle_3D(vec1, vec2))

                        if j == 0:
                            # MCP's second DOF: folding along X axis
                            vec1_xz = vec1.copy(); vec1_xz[1] = 0
                            vec2_xz = vec2.copy(); vec2_xz[1] = 0
                            finger_angles.append(self.calculate_angle(vec1_xz, vec2_xz))

                # === 3.2 Other fingers: index, middle, ring, pinky ===
                else:
                    bone = digit.bones[3]
                    tip_joint = bone.next_joint
                    tip_pos = self.get_joint_position(tip_joint, hand_position, quaternion)
                    fingers_pose.append(tip_pos[2])

                    for index_bone in range(4):
                        bone = digit.bones[index_bone]
                        start_data = self.get_joint_position(bone.prev_joint, hand_position, quaternion)
                        end_data   = self.get_joint_position(bone.next_joint, hand_position, quaternion)

                        if start_data and end_data:
                            start_xyz = np.array(start_data[2])
                            end_xyz   = np.array(end_data[2])

                            bone_vector = end_xyz - start_xyz
                            bone_vectors.append(bone_vector)

                            # Draw bone
                            cv2.line(self.output_image,
                                     (start_data[0], start_data[1]),
                                     (end_data[0], end_data[1]),
                                     self.hands_colour, 2)
                            cv2.circle(self.output_image,
                                       (end_data[0], end_data[1]),
                                       3, self.hands_colour, -1)

                    # --- Same: compute 3 joint angles for 3 bones ---
                    for j in range(len(bone_vectors) - 1):
                        vec1 = bone_vectors[j]
                        vec2 = bone_vectors[j + 1]

                        # YZ projection: typical flexion/extension
                        vec1_yz = vec1.copy(); vec1_yz[0] = 0
                        vec2_yz = vec2.copy(); vec2_yz[0] = 0
                        finger_angles.append(self.calculate_angle(vec1_yz, vec2_yz))

                        if j == 0:
                            # MCP additional abduction/adduction (XZ plane)
                            vec1_xz = vec1.copy(); vec1_xz[1] = 0
                            vec2_xz = vec2.copy(); vec2_xz[1] = 0
                            finger_angles.append(self.calculate_angle(vec1_xz, vec2_xz))

                # Add finger angles to whole hand list
                fingers_angles.append(finger_angles)

            # Convert angles to radians and reorder by finger
            reordered_fingers_angles = [
                [np.radians(row[0]), np.radians(row[1]), np.radians(row[2]), np.radians(row[3])]
                for row in fingers_angles
            ]
            #self.send_angles_udp(reordered_fingers_angles)  # Send UDP data
            print(fingers_pose)
            self.send_angles_udp(fingers_pose)

            # Display angles on image
            finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
            for idx, angles in enumerate(reordered_fingers_angles):
                angle_text = f"{finger_names[idx]} Angles: " + ', '.join([f"{angle:6.2f}" for angle in angles])
                cv2.putText(self.output_image, angle_text, (self.screen_size[1] - 600, text_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.font_colour, 1)
                text_offset += 20

# Define listener class to handle Leap Motion events
class TrackingListener(leap.Listener):
    def __init__(self, canvas):
        self.canvas = canvas

    def on_tracking_event(self, event):
        self.canvas.render_hands(event)  # Pass event to canvas for visualization

# Main function entry point
def main():
    canvas = Canvas()  # Create visual canvas
    tracking_listener = TrackingListener(canvas)  # Initialize listener
    connection = leap.Connection()  # Create Leap connection
    connection.add_listener(tracking_listener)  # Register listener

    # Start connection and set desktop tracking mode
    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)

        # Loop to continuously display hand skeleton
        while True:
            cv2.imshow(canvas.name, canvas.output_image)  # Show image
            if cv2.waitKey(1) == ord("x"):  # Press "x" to exit
                break

if __name__ == "__main__":
    main()
