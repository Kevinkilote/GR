#!/usr/bin/env python3
"""CARLA manual control with optional YOLOv11 live detection overlay."""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional, Tuple

import numpy as np
import pygame

import manual_control_steeringwheel as base


class DetectionContext:
    """Lazy loads a YOLO model and throttles inference for live overlays."""

    def __init__(
        self,
        weights_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
        min_interval: float = 0.1,
    ) -> None:
        self.weights_path = weights_path
        self.confidence = conf
        self.iou = iou
        self.device = device
        self.min_interval = max(0.0, min_interval)
        self.active = False
        self._model = None
        self._load_error: Optional[Exception] = None
        self._last_inference_time = 0.0
        self._last_result = None

    def toggle(self) -> Tuple[bool, Optional[Exception]]:
        """Toggle detection on/off, attempting to load the model on-demand."""
        if self.active:
            self.active = False
            return True, None
        try:
            self._ensure_model()
            self.active = True
            return True, None
        except Exception as exc:  # pylint: disable=broad-except
            self.active = False
            self._load_error = exc
            return False, exc

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError("ultralytics package is required for live detection") from exc
        self._model = YOLO(self.weights_path)
        return self._model

    def infer(self, frame_rgb: np.ndarray):
        if not self.active or self._model is None:
            return self._last_result
        now = time.time()
        if now - self._last_inference_time < self.min_interval:
            return self._last_result
        inference_kwargs = dict(conf=self.confidence, iou=self.iou, verbose=False)
        if self.device:
            inference_kwargs["device"] = self.device
        try:
            results = self._model(frame_rgb[:, :, ::-1], **inference_kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            logging.getLogger(__name__).error('YOLO inference failed: %s', exc)
            self._last_result = None
            self.active = False
            return None
        self._last_inference_time = now
        self._last_result = results[0] if results else None
        return self._last_result


class LiveDetectionCameraManager(base.CameraManager):
    """Camera manager that overlays YOLO detections when enabled."""

    BOX_COLOR = (0, 255, 0)
    BOX_WIDTH = 2
    BG_COLOR = (20, 20, 20)

    def __init__(self, parent_actor, hud, detection: DetectionContext):
        super().__init__(parent_actor, hud)
        self._detection = detection
        self._label_font = pygame.font.Font(pygame.font.get_default_font(), 18)
        self._class_names = None

    def set_sensor(self, index, notify=True):  # noqa: D401 (interface inherited)
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent,
            )
            weak_self = base.weakref.ref(self)
            self.sensor.listen(lambda image: LiveDetectionCameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        sensor_id = self.sensors[self.index][0]
        if sensor_id.startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data).astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img = np.zeros((self.hud.dim[0], self.hud.dim[1], 3))
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
            return

        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        rgb_array = array[:, :, ::-1]
        detection_result = self._detection.infer(rgb_array)
        surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        if detection_result and getattr(detection_result, "boxes", None):
            if self._class_names is None and hasattr(detection_result, "names"):
                self._class_names = detection_result.names
            self._draw_detections(surface, detection_result)
        self.surface = surface

    def _draw_detections(self, surface: pygame.Surface, detection_result) -> None:
        boxes = detection_result.boxes.xyxy.cpu().numpy()
        scores = detection_result.boxes.conf.cpu().numpy()
        classes = detection_result.boxes.cls.cpu().numpy().astype(int)
        width, height = surface.get_width(), surface.get_height()
        for bbox, score, cls_idx in zip(boxes, scores, classes):
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            rect = pygame.Rect(x1, y1, x2 - x1, y2 - y1)
            pygame.draw.rect(surface, self.BOX_COLOR, rect, self.BOX_WIDTH)
            label = f"{self._resolve_name(cls_idx)} {score:.2f}"
            self._draw_label(surface, rect, label)

    def _draw_label(self, surface: pygame.Surface, rect: pygame.Rect, label: str) -> None:
        text_surface = self._label_font.render(label, True, (255, 255, 255))
        text_bg = pygame.Surface((text_surface.get_width() + 6, text_surface.get_height() + 4))
        text_bg.fill(self.BG_COLOR)
        text_bg.blit(text_surface, (3, 2))
        surface.blit(text_bg, (rect.x, max(0, rect.y - text_surface.get_height() - 4)))

    def _resolve_name(self, cls_idx: int) -> str:
        if isinstance(self._class_names, dict) and cls_idx in self._class_names:
            return str(self._class_names[cls_idx])
        if isinstance(self._class_names, (list, tuple)) and 0 <= cls_idx < len(self._class_names):
            return str(self._class_names[cls_idx])
        return f"cls_{cls_idx}"


class LiveDetectionWorld(base.World):
    """World wrapper that swaps in a detection-aware camera manager."""

    def __init__(self, carla_world, hud, actor_filter, detection: DetectionContext):
        self._detection = detection
        super().__init__(carla_world, hud, actor_filter)

    def restart(self):
        super().restart()
        self._attach_detection_camera()

    def _attach_detection_camera(self) -> None:
        previous_manager = getattr(self, 'camera_manager', None)
        transform_index = 0
        sensor_index = 0
        if previous_manager is not None:
            transform_index = previous_manager.transform_index
            sensor_index = previous_manager.index or 0
            if previous_manager.sensor is not None:
                previous_manager.sensor.stop()
                previous_manager.sensor.destroy()
        self.camera_manager = LiveDetectionCameraManager(self.player, self.hud, self._detection)
        self.camera_manager.transform_index = transform_index
        self.camera_manager.set_sensor(sensor_index, notify=False)


class LiveDualControl(base.DualControl):
    """Extends DualControl to toggle live detection from input events."""

    def __init__(self, world, start_in_autopilot, detection: DetectionContext, detect_button: Optional[int]):
        self._detection = detection
        self._detect_button = detect_button if detect_button is None or detect_button >= 0 else None
        super().__init__(world, start_in_autopilot)

    def parse_events(self, world, clock):  # noqa: D401
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.JOYBUTTONDOWN:
                if self._detect_button is not None and event.button == self._detect_button:
                    self._handle_detection_toggle(world)
                    continue
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                if event.key == pygame.K_BACKSPACE:
                    world.restart()
                elif event.key == pygame.K_F1:
                    world.hud.toggle_info()
                elif event.key == pygame.K_h or (event.key == pygame.K_SLASH and pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == pygame.K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == pygame.K_c:
                    world.next_weather()
                elif event.key == pygame.K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > pygame.K_0 and event.key <= pygame.K_9:
                    world.camera_manager.set_sensor(event.key - 1 - pygame.K_0)
                elif event.key == pygame.K_r:
                    world.camera_manager.toggle_recording()
                elif event.key == pygame.K_l:
                    self._handle_detection_toggle(world)
                if isinstance(self._control, base.carla.VehicleControl):
                    if event.key == pygame.K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == pygame.K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('Manual Transmission' if self._control.manual_gear_shift else 'Automatic Transmission')
                    elif self._control.manual_gear_shift and event.key == pygame.K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == pygame.K_PERIOD:
                        self._control.gear += 1
                    elif event.key == pygame.K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, base.carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, base.carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _handle_detection_toggle(self, world):
        success, error = self._detection.toggle()
        if success and self._detection.active:
            world.hud.notification('Live detection ON')
        elif success:
            world.hud.notification('Live detection OFF')
        else:
            world.hud.notification(f'Live detection failed: {error}')


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    detection_context = DetectionContext(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        min_interval=args.detection_interval,
    )
    try:
        client = base.carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        hud = base.HUD(args.width, args.height)
        world = LiveDetectionWorld(client.get_world(), hud, args.filter, detection_context)
        controller = LiveDualControl(world, args.autopilot, detection_context, args.detect_button)
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
    finally:
        if world is not None:
            world.destroy()
        pygame.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CARLA manual control with optional YOLO live detection')
    parser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    parser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    parser.add_argument('--filter', metavar='PATTERN', default='vehicle.*', help='actor filter (default: "vehicle.*")')
    parser.add_argument('--weights', default='best.pt', help='path to YOLOv11 weights (default: best.pt)')
    parser.add_argument('--conf', default=0.25, type=float, help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--iou', default=0.45, type=float, help='YOLO IoU threshold (default: 0.45)')
    parser.add_argument('--device', default=None, help='Torch device for YOLO (e.g., cuda:0)')
    parser.add_argument('--detection-interval', default=0.1, type=float, help='minimum seconds between YOLO inferences (default: 0.1)')
    parser.add_argument('--detect-button', default=5, type=int, help='joystick button index to toggle detection (default: 5, set -1 to disable)')
    parser.add_argument('--debug', action='store_true', help='print debug information')
    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    if args.detect_button is not None and args.detect_button < 0:
        args.detect_button = None
    return args


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
