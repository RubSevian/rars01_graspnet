"""Persistent spawned YOLO worker. No ROS, camera or actuator ownership."""
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path

import numpy as np


def _worker_main(connection, config, root, classes, cpu_threads):
    try:
        import cv2
        import torch
        from utils.yolo_utils import load_yolo, detect_objects
        cv2.setNumThreads(cpu_threads)
        torch.set_num_threads(cpu_threads)
        model, options = load_yolo(config, project_root=Path(root), extra_classes=classes)
        image = np.zeros((config['camera']['color_height'], config['camera']['color_width'], 3), np.uint8)
        detect_objects(model, image, options)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        connection.send({'type': 'ready'})
        while True:
            request = connection.recv()
            if request['type'] == 'stop':
                return
            started = time.monotonic_ns()
            _, detections = detect_objects(model, request['image'], options)
            # NumPy masks and scalar metadata only, never CUDA tensors/Results.
            connection.send(dict(type='result', id=request['id'],
                capture_ns=request['capture_ns'], started_ns=started,
                completed_ns=time.monotonic_ns(), detections=detections))
            del request, detections, _
    except BaseException as error:
        try:
            connection.send(dict(type='error', error=f'{type(error).__name__}: {error}'))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class YoloWorker:
    def __init__(self, config, root, classes, cpu_threads, timeout_s, *, _target=_worker_main):
        self._closed = False
        self._close_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._sequence = 0
        context = mp.get_context('spawn')
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_target,
            args=(child, config, str(root), classes, cpu_threads),
            name='yolo-worker', daemon=True)
        self._process.start()
        child.close()
        try:
            message = self._receive(time.monotonic() + timeout_s, lambda: None)
            if message.get('type') != 'ready':
                raise RuntimeError(message.get('error', 'YOLO_STARTUP_FAILED'))
        except BaseException:
            self.close(force=True)
            raise

    def _receive(self, deadline, check_cancel):
        while True:
            check_cancel()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('YOLO_DEADLINE_EXCEEDED')
            if self._connection.poll(min(.02, remaining)):
                try:
                    return self._connection.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError('YOLO_WORKER_EXITED') from error
            if not self.is_alive():
                raise RuntimeError('YOLO_WORKER_EXITED')

    def detect(self, image, capture_ns, deadline, check_cancel):
        # No queued frames and no second request while the worker is occupied.
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError('YOLO_BUSY')
        try:
            check_cancel()
            if not self.is_alive():
                raise RuntimeError('YOLO_WORKER_EXITED')
            if time.monotonic() >= deadline:
                raise TimeoutError('YOLO_DEADLINE_EXCEEDED')
            self._sequence += 1
            # Pipe.send/recv may block halfway through a large frame/mask.
            # Keep both inside a disposable transport thread so the caller can
            # enforce cancellation/deadline even if the child stops reading.
            result = queue.Queue(maxsize=1)
            request = dict(type='detect', id=self._sequence,
                capture_ns=capture_ns, image=np.ascontiguousarray(image))
            def exchange():
                try:
                    self._connection.send(request)
                    result.put((True, self._connection.recv()))
                except (EOFError, OSError) as error:
                    result.put((False, error))
            threading.Thread(target=exchange, name='yolo-ipc', daemon=True).start()
            while True:
                check_cancel()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('YOLO_DEADLINE_EXCEEDED')
                try:
                    ok, message = result.get(timeout=min(.02, remaining))
                    break
                except queue.Empty:
                    if not self.is_alive():
                        raise RuntimeError('YOLO_WORKER_EXITED')
            if not ok:
                raise RuntimeError('YOLO_WORKER_EXITED') from message
            if time.monotonic() >= deadline:
                raise TimeoutError('YOLO_DEADLINE_EXCEEDED')
            check_cancel()
            if message.get('type') == 'error':
                raise RuntimeError(message['error'])
            if (message.get('type') != 'result' or message.get('id') != self._sequence
                    or message.get('capture_ns') != capture_ns):
                raise RuntimeError('YOLO_RESPONSE_MISMATCH')
            return message
        except BaseException:
            # No unread late result may become the next request's response.
            self.close(force=True)
            raise
        finally:
            self._request_lock.release()

    def is_alive(self):
        return not self._closed and self._process.is_alive()

    def close(self, force=False):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._process.is_alive() and not force:
                try:
                    self._connection.send({'type': 'stop'})
                    self._process.join(timeout=1.)
                except (EOFError, OSError):
                    pass
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2.)
            self._connection.close()
