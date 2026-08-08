# RARS01 GraspNet

Рабочий проект захвата объектов роботом RARS01 с камерой Orbbec Gemini 336,
YOLOE и GraspNet. Камера установлена на руке; положение камеры относительно
`End_link` определяется eye-in-hand калибровкой ArUco без изменения URDF.

Основной рабочий сценарий перенесён из `reBot-DevArm-Grasp` и адаптирован под:

- семь моторов RARS SDK (шесть суставов и гриппер);
- неизменённый `rars01.urdf`;
- Orbbec Gemini 336 с автоматическим fallback с hardware D2C на software align;
- текущий гриппер RARS01, раскрывающийся в плоскости `XZ`;
- выбор захвата `graspnet` или `central_mask`;
- вертикальный заход сверху или исходный заход по лучу камеры;
- исходную minimum-jerk последовательность движений проекта reBot.

Будущая замена механики гриппера на конструкцию reBot описана в
[`FUTURE_REBOT_GRIPPER.md`](FUTURE_REBOT_GRIPPER.md).

## Структура

```text
calibration/                 Python-код hand-eye/Aruco
config/default.yaml          основной конфиг робота и захвата
config/calibration/          intrinsics и текущая hand-eye матрица
drivers/camera/              Gemini 336 и RealSense
drivers/robot/               адаптер RARS01 к алгоритмам reBot и гриппер
models/                      локальные веса YOLO и GraspNet
scripts/grasp.py             основной запуск захвата
scripts/collect_handeye_eih.py автоматическая eye-in-hand калибровка
sdk/graspnet-baseline/       локально установленная сторонняя библиотека
utils/                       YOLO, GraspNet, преобразования и визуализация
```

Каталог `.venv` уже является окружением этого проекта. Его не копируют в Git и
не переносят между компьютерами. На существующей машине продолжайте использовать
его через `uv run`; `uv.lock` фиксирует Python-зависимости.

## Внешние репозитории

Они не копируются внутрь данного Git-репозитория:

1. `rars_arm_sdk` — обмен с моторами и модуль `rars_arm_py`;
2. `rars01_description` — `urdf/rars01.urdf` и meshes;
3. `reBotArm_control_py` — Pinocchio IK и исходный Cartesian/minimum-jerk
   контроллер;
4. `graspnet-baseline` и `graspnetAPI` — нейросеть и API GraspNet.

Текущая рабочая раскладка:

```text
RARS_sdk_grasp_net/
├── rars01_graspnet/
├── rars_arm_sdk/
├── rars01_description/
└── reBot-DevArm-Grasp/sdk/reBotArm_control_py/
```

Драйвер автоматически ищет reBot-контроллер также в переносимом варианте
`rars01_graspnet/sdk/reBotArm_control_py` и в соседнем
`../reBotArm_control_py`. Явный путь можно задать в
`robot.repo_root` файла `config/default.yaml`.

## Существующее окружение

Проверка без переустановки:

```bash
cd /home/ruben/RARS_sdk_grasp_net/rars01_graspnet
uv run python --version
uv run python scripts/grasp.py --help
uv run python -m pytest -q
```

Для нового компьютера:

```bash
uv python install 3.10
uv sync --python 3.10 --extra camera --extra robot --extra vision --extra dev
```

После этого установите reBot-контроллер в то же окружение, например для текущей
структуры workspace:

```bash
uv pip install -e ../reBot-DevArm-Grasp/sdk/reBotArm_control_py
```

## RARS SDK

Модуль собирается из соседнего репозитория непосредственно для `.venv`:

```bash
cmake -S ../rars_arm_sdk -B ../rars_arm_sdk/build-python \
  -DRARS_ARM_BUILD_QT_EXAMPLE=OFF \
  -DRARS_ARM_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(uv run python -m pybind11 --cmakedir)"
cmake --build ../rars_arm_sdk/build-python -j
uv run python -c "import sys; sys.path.insert(0, '../rars_arm_sdk/build-python'); import rars_arm_py; print('RARS SDK OK')"
```

Пути к собранному модулю и URDF уже заданы относительно корня проекта:

```yaml
robot:
  rars01:
    sdk_python_path: ../rars_arm_sdk/build-python
    urdf_path: ../rars01_description/urdf/rars01.urdf
```

## Orbbec Gemini 336

Используется официальный Python-модуль `pyorbbecsdk`. Режим
`camera.alignment: auto` сначала ищет совместимый hardware D2C профиль, а при
ошибке `Current stream profile is not support hardware d2c process` включает
официальный software `AlignFilter`.

Проверка камеры:

```bash
uv run python scripts/grasp.py --help
```

Для реального preview используйте основной `scripts/grasp.py`; отдельные старые
диагностические scripts сохранены для проверки компонентов.

## Модели

В `models/` находятся проверенные локальные файлы:

```text
yolo11n.pt
yolov8s-world.pt
yoloe-26s-seg.pt
yoloe-26l-seg.pt       основной YOLOE из config/default.yaml
checkpoint-rs.tar      GraspNet epoch 18
```

YOLO-веса поэтому повторно не скачиваются. При первом вызове
`YOLOE.set_classes()` Ultralytics может автоматически скачать дополнительный
текстовый encoder `mobileclip2_b.ts` (~242 МБ) в корень проекта. Файл добавлен
в `.gitignore`; после первого запуска он остаётся локально.

GraspNet **не скачивает** `checkpoint-rs.tar` автоматически. Проверенная копия
лежит в `models/`, а путь задан как:

```yaml
graspnet:
  checkpoint: models/checkpoint-rs.tar
```

Целостность всех перенесённых весов проверяется командой:

```bash
cd models && sha256sum -c SHA256SUMS
```

Исходники GraspNet и CUDA extensions устанавливаются отдельно:

```bash
mkdir -p sdk
git clone --depth 1 https://github.com/graspnet/graspnet-baseline.git \
  sdk/graspnet-baseline
git clone --depth 1 https://github.com/graspnet/graspnetAPI.git \
  sdk/graspnet-baseline/graspnetAPI

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
bash scripts/install_graspnet.sh
uv run python scripts/check_graspnet.py
```

Версия `nvcc` должна совпадать с CUDA-сборкой PyTorch. На проверенной машине:

```text
PyTorch 2.13.0+cu130
CUDA 13.0
NVIDIA GeForce RTX 4080 SUPER
```

## Калибровка

Поставьте руку в нулевое положение, закрепите ArUco неподвижно на столе и
освободите область движения:

```bash
uv run python scripts/collect_handeye_eih.py --robot-backend rars01
```

Введите `START`. Скрипт автоматически обходит калибровочные позы, сохраняет
доступные наблюдения и возвращает руку домой. Результат:

```text
config/calibration/orbbec_gemini2/hand_eye.npz
```

Камера не обязана видеть маркер в home — он должен быть виден хотя бы в
достаточном количестве калибровочных поз.

## Захват банана

Сначала безопасный dry-run:

```bash
uv run python scripts/grasp.py \
  --robot-backend rars01 \
  --target-class banana \
  --dry-run
```

Dry-run всё равно переводит руку `home → ready`, но не выполняет найденный
захват. Для настоящего захвата:

```bash
uv run python scripts/grasp.py \
  --robot-backend rars01 \
  --target-class banana
```

Управление:

```text
G / Space   распознать объект и выполнить захват
R           вернуться к live preview
Q / Esc     освободить объект, вернуться home и отключить моторы
```

Основные режимы выбираются в `config/default.yaml`:

```yaml
grasp_pipeline:
  mode: central_mask            # central_mask или graspnet
  grasp:
    central_mask_approach: vertical  # vertical или camera_ray
```

Последовательность и длительности оставлены как в исходном рабочем сценарии
reBot: `pregrasp=2.0 с`, `grasp=1.5 с`, `retreat=1.5 с`, `ready=3.0 с`.
Дополнительных пауз и ожидания стабилизации между участками нет.

## Безопасность

- Перед `START` рука должна физически находиться около нулевых углов.
- Первый запуск после изменения конфигурации выполняйте без объекта и с рукой
  над свободным столом.
- Не увеличивайте `KP`, момент или частоту до проверки обратной связи.
- При ошибке основной сценарий пытается освободить гриппер, вернуть руку домой
  и отключить моторы.
- Реальные calibration `.npz` относятся только к данной установке камеры и
  робота.
