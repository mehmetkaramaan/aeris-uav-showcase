import math, sys, time, threading
from pathlib import Path
import numpy as np
import cv2
import os
if os.environ.get('AERIS_GZ_PYTHONPATH'):
    sys.path.append(os.environ['AERIS_GZ_PYTHONPATH'])
from gz.transport13 import Node, SubscribeOptions
from gz.msgs10.image_pb2 import Image
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.stringmsg_pb2 import StringMsg
from avoidance import Avoidance
from perception import landmarks
import json

class Vision:

    def __init__(self, telemetry):
        self.clock_time = 0.0
        self.clock_last = 0.0
        self.safety = Avoidance()
        self.front = None
        self.front_last = 0
        self.front_boxes = []
        self.front_landmarks = []
        self.traffic = {}
        self.traffic_pub = None
        self.telemetry = telemetry
        self.lock = threading.RLock()
        self.node = Node()
        self.raw = None
        self.jpg = None
        self.filtered = None
        self.last = 0.0
        self.count = 0
        self.filter = 'none'
        self.noise = 0
        self.detections = []
        self.line = None
        self.truth = None
        opt = SubscribeOptions()
        opt.msgs_per_sec = 10
        assert self.node.subscribe(Image, '/aeris/camera', self.callback, opt)
        self.node.subscribe(Pose_V, '/world/aeris/dynamic_pose/info', self.poses)
        self.node.subscribe(Clock, '/world/aeris/clock', self.clock_callback)
        self.node.subscribe(LaserScan, '/aeris/lidar', self.lidar)
        self.node.subscribe(Image, '/aeris/front', self.front_callback, opt)
        self.node.subscribe(StringMsg, '/aeris/traffic/state', self.traffic_callback)
        self.traffic_pub = self.node.advertise('/aeris/traffic/control', StringMsg)

    def clock_callback(self, msg):
        value = msg.sim.sec + msg.sim.nsec / 1000000000.0
        if value > self.clock_time:
            self.clock_time = value
            self.clock_last = time.monotonic()

    def lidar(self, msg):
        try:
            self.safety.update(msg, self.telemetry())
        except Exception as e:
            print('LiDAR:', e, flush=True)

    def traffic_callback(self, msg):
        try:
            self.traffic = json.loads(msg.data)
        except ValueError:
            pass

    def configure_traffic(self, data):
        msg = StringMsg()
        msg.data = json.dumps(data)
        self.traffic_pub.publish(msg)

    def front_callback(self, msg):
        try:
            frame = cv2.cvtColor(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step or msg.width * 3)[:, :msg.width * 3].reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
            self.front_landmarks = landmarks(frame, self.telemetry(), self.project, front=True)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            mask = ((h > 90) & (h < 165) & (s > 120) & (v > 50) | (h > 20) & (h < 38) & (s > 120) & (v > 80)).astype('uint8') * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in contours:
                if cv2.contourArea(c) < 16:
                    continue
                x, y, w, hh = cv2.boundingRect(c)
                boxes.append([x, y, w, hh])
                cv2.rectangle(frame, (x, y), (x + w, y + hh), (255, 230, 40), 2)
                cv2.putText(frame, 'Trafik adayi', (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 230, 40), 1)
            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.front = jpg.tobytes()
            self.front_last = time.monotonic()
            self.front_boxes = boxes
        except Exception as e:
            print('Front camera:', e, flush=True)

    def poses(self, msg):
        for p in msg.pose:
            if p.name == 'aeris_x500_0':
                self.truth = (p.position.x, p.position.y, p.position.z)

    def project(self, u, v, height, st):
        f = 640 / (2 * math.tan(1.4 / 2))
        ray = np.array([(240 - v) / f, (u - 320) / f, 1.0])
        cr, sr = (math.cos(st.get('roll', 0)), math.sin(st.get('roll', 0)))
        cp, sp = (math.cos(st.get('pitch', 0)), math.sin(st.get('pitch', 0)))
        cy, sy = (math.cos(st.get('yaw', math.pi / 2)), math.sin(st.get('yaw', math.pi / 2)))
        R = np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy], [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy], [-sp, sr * cp, cr * cp]])
        north, east, down = R @ ray
        scale = height / max(0.2, down)
        return {'east': st.get('east', 0) + east * scale, 'north': st.get('north', 0) + north * scale}

    def callback(self, msg):
        try:
            step = msg.step or msg.width * 3
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if self.noise:
                frame = np.clip(frame.astype(np.float32) + np.random.default_rng().normal(0, self.noise, frame.shape), 0, 255).astype(np.uint8)
            raw = frame.copy()
            if self.filter == 'gaussian':
                frame = cv2.GaussianBlur(frame, (5, 5), 1.1)
            elif self.filter == 'median':
                frame = cv2.medianBlur(frame, 5)
            elif self.filter == 'sharpen':
                frame = cv2.addWeighted(frame, 1.8, cv2.GaussianBlur(frame, (5, 5), 1), -0.8, 0)
            st = self.telemetry()
            dets = landmarks(frame, st, self.project)
            line = None
            cv2.drawMarker(frame, (320, 240), (220, 220, 220), cv2.MARKER_CROSS, 16, 1)
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            _, rjpg = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 78])
            with self.lock:
                self.last = time.monotonic()
                self.count += 1
                self.raw = rjpg.tobytes()
                self.jpg = jpg.tobytes()
                self.line = line
                self.detections = dets
        except Exception as e:
            print('Camera frame error:', e, flush=True)

    def state(self):
        with self.lock:
            return {'sim_time': self.clock_time, 'sim_age': time.monotonic() - self.clock_last if self.clock_last else None, 'age': round(time.monotonic() - self.last, 2) if self.last else None, 'frames': self.count, 'line': self.line, 'detections': list(self.detections), 'filter': self.filter, 'noise': self.noise, 'front_age': round(time.monotonic() - self.front_last, 2) if self.front_last else None, 'front_candidates': len(self.front_boxes), 'front_landmarks': list(self.front_landmarks)}

    def move_model(self, name, x, y, z):
        p = Pose()
        p.name = name
        p.position.x = x
        p.position.y = y
        p.position.z = z
        p.orientation.w = 1
        return self.node.request('/world/aeris/set_pose', p, Pose, Boolean, 1000)
