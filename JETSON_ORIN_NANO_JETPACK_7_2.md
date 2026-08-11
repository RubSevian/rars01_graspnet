# RARS01 GraspNet на Jetson Orin Nano / JetPack 7.2

Этот документ описывает целевую архитектуру переноса `rars01_graspnet` на
Jetson Orin Nano с JetPack 7.2. Цель — запускать тот же захват RGB-D:
Orbbec Gemini 336 → YOLO → GraspNet → IK → RARS01, не переписывая геометрию
гриппера, URDF, hand-eye калибровку и логику выбора захвата.

JetPack 7.2 содержит Jetson Linux 39.2, Ubuntu 24.04, CUDA 13.2.1 и TensorRT
10.16.2. См. [NVIDIA JetPack 7.2](https://developer.nvidia.com/embedded/jetpack/downloads).

## 1. Что переносится, а что остаётся прежним

Остаётся прежним:

- прошивка STM32, протокол USB CDC и настройки моторов;
- RARS01 URDF, `End_link`, ограничения суставов и геометрия гриппера;
- матрица hand-eye, если камера физически не переносилась;
- GraspNet, YOLO, преобразование camera → base, IK и minimum-jerk;
- YAML-параметры захвата, включая классы YOLO.

Нужно собрать заново под `aarch64`:

- виртуальное окружение Python;
- `rars_arm_py` из `rars_arm_sdk`;
- PyTorch для Jetson;
- CUDA-расширения GraspNet: `pointnet2` и `knn`;
- Pinocchio и локальный модуль `rars01_graspnet/rebot_math.py`;
- Orbbec Python SDK и udev rules.

Нельзя копировать с ПК `.venv` и собранные `.so`: они собраны для `x86_64`
и RTX 4080 (`sm_89`), а Orin Nano использует `aarch64` и CUDA-архитектуру
`sm_87`.

## 2. Архитектура исполнения

```text
Gemini 336 (USB 3)
       │ RGB + Depth + alignment
       ▼
Camera worker
       │
       ▼
Perception worker (GPU)
YOLOE mask → point cloud → GraspNet
       │ grasp candidates in camera frame
       ▼
Coordinator
hand-eye / camera→base → grasp selection → Pinocchio IK
       │ target trajectory
       ▼
Robot-control worker (100 Hz)
minimum-jerk → POS/VEL joints 1..6 + MIT gripper 7
       │ USB CDC
       ▼
STM32 → RARS01 motors
```

### Разделение ответственности

`Camera worker` владеет SDK камеры и выдаёт согласованные RGB-D кадры.
Для Gemini 336 сохраняется текущая стратегия: сначала hardware D2C, при
неподдерживаемом профиле — программный `AlignFilter`.

`Perception worker` использует GPU, но не отправляет команды моторам. Он
выполняет YOLO, строит облако точек и запускает GraspNet только по запросу
пользователя/координатора.

`Coordinator` хранит состояние сценария:

```text
HOME → READY → CAPTURE → INFER → PLAN → PREGRASP → GRASP
     → CLOSE → STABILIZE → RETREAT → READY/HOME
                         └→ FAULT → HOME/DISABLE
```

`Robot-control worker` должен продолжать посылать hold-команды, пока GPU
занят. Это не даёт inference задержке вызвать watchdog STM32. На первом
этапе допустим существующий выделенный control thread; для постоянной работы
на Jetson предпочтителен отдельный процесс с локальной очередью команд.

Эта граница позже прямо превращается в ROS 2-ноды: камера, perception,
grasp-execution и RARS hardware interface.

## 3. Частоты и режимы управления

| Часть | Режим / частота | Назначение |
|---|---:|---|
| Суставы 1–6 | POS/VEL, 100 Hz | Плавно выполняют точки Cartesian minimum-jerk. |
| Гриппер 7 | MIT, 100 Hz | Закрытие по моменту и удержание объекта. |
| Планировщик | `dt = 1 / 100 s` | Выдаёт точки траектории без повторов. |
| STM32 watchdog | 250 ms | Независимая защита при потере команд. |
| Камера | 30 FPS | Не является частотой моторного управления. |

На Jetson не следует сразу выставлять 500 Hz. Сначала нужна стабильная работа
100 Hz. Переход к 200, затем к 500 Hz имеет смысл только после измерения
джиттера USB CDC и загрузки CPU. Для настоящих 500 Hz надо одновременно
изменять `command_rate_hz` и `dt` планировщика на `0.002`, иначе будут
повторяться одинаковые точки.

## 4. Python, CUDA и зависимости

Текущий `pyproject.toml` ограничен CPython 3.10 и содержит desktop-зависимости
PyTorch. На Jetson нужно отдельное platform-specific окружение; `uv.lock` от
ПК использовать нельзя.

Рекомендуемая структура зависимостей:

```text
pyproject.toml              общие pure-Python зависимости
requirements/
├── common.txt              NumPy, YAML, OpenCV, SciPy и др.
├── desktop-cu130.txt       x86_64 / RTX разработка
└── jetson-jp72.txt         aarch64 / JetPack 7.2
```

Порядок установки Jetson:

1. Создать чистое виртуальное окружение на самом Jetson.
2. Установить Jetson-совместимый wheel PyTorch, соответствующий фактическому
   JetPack, CUDA и версии Python.
3. Установить остальные Python-зависимости без подмены PyTorch обычным wheel
   из PyPI.
4. Собрать нативные C++/CUDA модули.

NVIDIA публикует отдельные PyTorch-пакеты для Jetson; wheel надо выбирать по
матрице совместимости, а не по версии PyTorch на desktop.
См. [официальную инструкцию NVIDIA](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

Перед сборкой GraspNet необходимо задать:

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.7
```

Скрипт `scripts/install_graspnet.sh` должен определять эти значения для
Jetson автоматически и не использовать desktop-значения `cuda-13.0` и `8.9`.

## 5. Камера Orbbec

`pyorbbecsdk2` поддерживает Linux ARM64 и заявляет тестирование на Jetson
Orin Nano. См. [Orbbec pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk).

На Jetson требуется:

1. Установить OrbbecSDK v2 / `pyorbbecsdk2` для `aarch64`.
2. Выполнить установку udev rules из поставки SDK.
3. Подключить Gemini 336 к USB 3 и проверить топологию `lsusb -t`.
4. Проверить RGB, depth и fallback D2C отдельно от GraspNet.

## 6. Производительность и память Orin Nano

Первый рабочий профиль не должен одновременно удерживать тяжёлые модели в
GPU-памяти без измерений. Базовая конфигурация для старта:

```yaml
yolo:
  model_name: yoloe-26s-seg.pt
  device: cuda:0

graspnet:
  num_point: 10000
  top_k: 20
```

Также:

- запускать YOLO и GraspNet последовательно;
- отключить Open3D GUI на Jetson (`--no-open3d`);
- оставить кадр камеры с уже проверенным профилем, а не менять его сразу;
- после рабочего запуска измерить FPS и память, затем решить, нужны ли
  FP16, TensorRT для YOLO или большее число точек GraspNet.

TensorRT — оптимизация второго этапа. GraspNet CUDA-операторы остаются
PyTorch/CUDA и должны быть сначала проверены в обычном PyTorch запуске.

## 7. Модели

Робот не должен зависеть от интернета во время работы. В `models/` должны
быть подготовлены и проверены SHA256:

- `checkpoint-rs.tar` — checkpoint GraspNet;
- `yoloe-26s-seg.pt` — стартовая YOLOE для Orin;
- при необходимости `yoloe-26l-seg.pt`;
- `mobileclip2_b.ts`, если его требует выбранная YOLOE.

Нужен отдельный скрипт `scripts/download_models.sh` или
`scripts/download_models.py`: он скачивает отсутствующие файлы один раз и
проверяет `models/SHA256SUMS`. GraspNet checkpoint не следует рассчитывать
скачать неявно во время захвата.

## 8. Этапы переноса и проверки

### Этап A — система

1. Установить JetPack 7.2 на NVMe, обеспечить питание и охлаждение.
2. Проверить `uname -m` (`aarch64`), `nvcc --version`, `nvidia-smi`/`tegrastats`.
3. Проверить доступную RAM, swap/zram и USB 3.

### Этап B — робот без зрения

1. Собрать `rars_arm_sdk` на Jetson.
2. Проверить import `rars_arm_py`.
3. Проверить feedback всех семи моторов.
4. Выполнить безопасное движение в home/ready без камеры.
5. Проверить, что arm 1–6 в POS/VEL, гриппер в MIT, а watchdog работает.

### Этап C — камера без робота

1. Запустить Gemini 336.
2. Проверить цвет, глубину, alignment и стабильность 30 FPS.
3. Проверить сохранённый RGB-D кадр и его intrinsics.

### Этап D — модели

1. Установить PyTorch Jetson и проверить `torch.cuda.is_available()`.
2. Запустить YOLOE на сохранённом кадре.
3. Собрать `pointnet2` и `knn` с `TORCH_CUDA_ARCH_LIST=8.7`.
4. Запустить GraspNet на сохранённом кадре.

### Этап E — полный сценарий

1. Запустить `grasp.py --dry-run --no-open3d`.
2. Проверить `T_grasp_camera`, `T_grasp_base`, кандидаты IK и траекторию.
3. Проверить реальное движение без объекта.
4. Проверить медленный захват простого объекта.
5. Только затем повышать нагрузку, FPS или размер моделей.

## 9. Планируемые файлы для Jetson

```text
config/jetson_orin_nano.yaml    Jetson overrides: модель, num_point, Open3D
scripts/setup_jetson_72.sh      проверка системы и установка зависимостей
scripts/check_jetson.py         CUDA, torch, sm_87, RAM, camera, SDK, extensions
scripts/download_models.py      загрузка и SHA256-проверка моделей
JETSON_ORIN_NANO_JETPACK_7_2.md этот документ
```

До появления этих файлов запуск выполняется по этапам выше. Скрипты должны
быть добавлены после получения Jetson, чтобы зафиксировать фактические версии
JetPack, Python и NVIDIA PyTorch wheel, установленные на устройстве.

## 10. ROS 2

JetPack 7.2 основан на Ubuntu 24.04, поэтому нативная ROS 2-ветка — Jazzy.
Humble официально ориентирован на Ubuntu 22.04 ARM64. Если обязательно нужен
Humble, потребуется отдельный Ubuntu 22.04 контейнер или сборка из исходников.

Ссылки: [ROS 2 Jazzy для Ubuntu 24.04 ARM64](https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Install-Binary.html),
[поддерживаемые платформы Humble](https://docs.ros.org/en/humble/Releases/Release-Humble-Hawksbill.html).

Сначала следует обеспечить нативный standalone-запуск. После этого worker
границы из раздела 2 можно перенести в ROS 2 без изменения GraspNet, IK,
калибровки и протокола STM32.
