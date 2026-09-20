# RARS01 GraspNet: Jetson Orin Nano 8 GB / JetPack 7.2

Порядок подготовки нативного запуска на Jetson. Выполняйте этапы строго по
порядку. Не запускайте реальное движение робота до этапа 7.

Профиль запуска: `config/jetson_orin_nano.yaml`. Он использует
`yoloe-26s-seg.pt`, 10 000 точек GraspNet, `top_k: 20` и оставляет управление
роботом на 100 Hz.

## 0. Что должно быть рядом

```text
RARS_sdk_grasp_net/
├── rars01_graspnet/
├── rars_arm_sdk/
└── rars01_description/
```

Также потребуются:

- веса из `rars01_graspnet/models/`;
- ARM64 OrbbecSDK v2 / `pyorbbecsdk2` для Gemini 336;
- подключённые Gemini 336 и STM32 RARS01.

Проверка весов:

```bash
cd /home/ruben/RARS_sdk_grasp_net/rars01_graspnet/models
sha256sum -c SHA256SUMS
```

## 1. Система и память

Проверьте JetPack и CUDA:

```bash
cat /etc/nv_tegra_release
/usr/local/cuda-13.2/bin/nvcc --version
```

Установите системные инструменты:

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3.12-venv libopenblas-dev \
  libusb-1.0-0 libgl1 libglib2.0-0
```

Для CUDA-сборки на Orin Nano 8 GB нужен swap. Создайте 16 GB swap на NVMe:

```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
swapon --show
```

Если файл `/swapfile` уже существует, не выполняйте команды создания повторно;
проверьте его через `swapon --show`.

## 2. Python и CUDA PyTorch

```bash
cd /home/ruben/RARS_sdk_grasp_net/rars01_graspnet
uv venv --python 3.12

export UV_CACHE_DIR=/tmp/rars01-uv-cache
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=10
export UV_CONCURRENT_DOWNLOADS=1

uv pip install --python .venv/bin/python \
  'torch==2.13.0+cu132' 'torchvision==0.28.0+cu132' \
  --index-url https://download.pytorch.org/whl/cu132

uv pip install --python .venv/bin/python -e .
```

Проверка GPU:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```

Ожидается `True` и `(8, 7)`. Upstream wheel может выдать предупреждение про
SM 8.7; базовые CUDA tensor и `Conv2d` на этом Jetson проверены.

## 3. Остальные Python-пакеты

```bash
uv pip install --python .venv/bin/python --no-deps \
  'scipy==1.18.0' transforms3d trimesh ninja pybind11 pytest \
  matplotlib pillow psutil py-cpuinfo pandas requests tqdm ultralytics-thop \
  contourpy cycler fonttools kiwisolver packaging pyparsing python-dateutil \
  pytz tzdata charset-normalizer idna urllib3 certifi
uv pip install --python .venv/bin/python --no-deps 'ultralytics>=8.4,<8.5'
```

YOLOE использует текстовый encoder CLIP для классов из `yolo.custom_classes`.
Установите официальный вариант Ultralytics (один раз):

```bash
git clone --depth 1 https://github.com/ultralytics/CLIP.git sdk/ultralytics-clip
uv pip install --python .venv/bin/python --no-deps ./sdk/ultralytics-clip
```

Для `yoloe-26s-seg.pt` также нужен вес `mobileclip2_b.ts` (242 МБ). Старая
версия Ultralytics не умеет скачать его автоматически, поэтому загрузите его
в корень проекта до первого запуска:

```bash
curl -L --fail --continue-at - -o mobileclip2_b.ts \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts
```

`open3d` не устанавливайте: для Linux ARM64 + Python 3.12 нет готового колеса.
Скрипт установки GraspNet отключает его обязательный импорт; инференс и
`--no-open3d` работают без него. Open3D-визуализация GraspNet на Jetson
недоступна.

## 4. Orbbec Gemini 336

1. Установите предоставленный Orbbec ARM64 пакет в `.venv`, не давая ему
   обновить закреплённый NumPy:

```bash
uv pip install --python .venv/bin/python --no-deps --upgrade pyorbbecsdk2
uv pip install --python .venv/bin/python --no-deps --reinstall 'numpy==2.3.5'
```

2. Установите udev rules официальным скриптом OrbbecSDK:

```bash
git clone --depth 1 https://github.com/orbbec/OrbbecSDK.git /tmp/OrbbecSDK
cd /tmp/OrbbecSDK/misc/scripts
sudo chmod +x ./install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
```

3. Переподключите камеру к USB 3.

Проверка после установки SDK:

```bash
.venv/bin/python scripts/check_camera.py --config config/jetson_orin_nano.yaml
```

## 5. Pinocchio и RARS SDK

Контроллер и математика входят в проект. Установите нативные зависимости:

```bash
uv pip install --python .venv/bin/python \
  'numpy==2.3.5' 'scipy==1.18.0' 'pin==3.9.0' \
  'cmeel-urdfdom==4.0.1' 'cmeel-tinyxml2==10.0.0'
```

Перед любым запуском, который использует робота, добавьте библиотеки Pinocchio
в путь динамического загрузчика (выполняется в каждом новом терминале):

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/cmeel.prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Внешний Python-пакет управления рукой не требуется; `robot.repo_root` удалён.

Соберите модуль RARS SDK:

```bash
cmake -S ../rars_arm_sdk -B ../rars_arm_sdk/build-python \
  -DRARS_ARM_BUILD_QT_EXAMPLE=OFF -DRARS_ARM_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build ../rars_arm_sdk/build-python -j2
```

Проверка без движения:

```bash
.venv/bin/python scripts/check_robot.py --config config/jetson_orin_nano.yaml
```

### Стабильный порт STM32 после холодного старта

Драйвер ждёт `/dev/ttyACM0` до 20 секунд и повторяет подключение без включения
моторов. Чтобы номер `ttyACM` не менялся, один раз создайте постоянное имя:

```bash
sudo install -m 644 config/udev/99-rars01-stm32.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

После переподключения STM32 убедитесь, что есть `/dev/rars01_stm32`, и замените
`robot.rars01.port` в YAML на `/dev/rars01_stm32`. Если устройства нет даже в
`lsusb`, Type-C не перешёл в USB-host или есть аппаратная проблема: ожидание
порта это исправить не может.

Если STM32 есть в `lsusb`, но `/dev/ttyACM*` не появился или порт завис,
выключите моторы и переинициализируйте только STM32 (Gemini не затрагивается):

```bash
sudo .venv/bin/python scripts/reset_stm32_usb.py --execute
```

## 6. GraspNet CUDA-операторы

Исходники уже должны быть в `sdk/graspnet-baseline/` и
`sdk/graspnet-baseline/graspnetAPI/`. Если их нет:

```bash
git clone --depth 1 https://github.com/graspnet/graspnet-baseline.git sdk/graspnet-baseline
git clone --depth 1 https://github.com/graspnet/graspnetAPI.git sdk/graspnet-baseline/graspnetAPI
```

Собирайте строго с одним job, чтобы не перегрузить Jetson и VS Code:

```bash
export CUDA_HOME=/usr/local/cuda-13.2
export TORCH_CUDA_ARCH_LIST=8.7
export MAX_JOBS=1
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/install_graspnet.sh
```

Проверка:

```bash
.venv/bin/python scripts/check_graspnet.py --config config/jetson_orin_nano.yaml
```

## 7. Безопасный первый запуск

На Jetson GraspNet запускается в отдельном процессе `graspnet-cuda-worker`.
Модель загружается до подключения робота. После нажатия `G` основной процесс
ожидает ответ worker, а существующий цикл RARS01 продолжает отправлять текущую
позицию с частотой 100 Hz. Камера и `/dev/ttyACM0` остаются только в основном
процессе; worker не имеет доступа к железу. Максимальное ожидание задаётся
`graspnet.worker_timeout_s` (по умолчанию 30 секунд).

Пересобирать `rars_arm_sdk` для этого режима не требуется.

Сначала камера и робот без объекта, с рукой над свободным столом:

```bash
.venv/bin/python scripts/check_jetson.py
.venv/bin/python scripts/grasp.py \
  --config config/jetson_orin_nano.yaml --dry-run --no-open3d
```

Только затем проверяйте реальное движение без объекта, а после этого — медленный
захват простого объекта. Не повышайте частоту 100 Hz и не включайте TensorRT до
стабильной работы камеры, USB CDC и GraspNet.

При нажатии `G` кандидаты GraspNet сортируются по score, для параллельного
гриппера проверяются обе эквивалентные ориентации, а IK решается цепочкой
`текущая поза -> pregrasp -> grasp -> retreat`. RARS01 выполняет уже найденные
суставные решения в POS_VEL, не запуская IK повторно.

Формулы выбора наиболее удобного и безопасного кандидата: [GRASP_SELECTION_MATH_CODEX.md](GRASP_SELECTION_MATH_CODEX.md).

Постоянное смещение после проверки hand-eye задаётся в `config/default.yaml`:

```yaml
grasp_pipeline:
  grasp:
    position_compensation_base_m: {x: 0.0, y: 0.0, z: 0.0}
```

Значения задаются в метрах по осям `base_link`. Если губки стабильно приходят,
например, на `+5 мм` дальше по X, установить `x: -0.005`. Менять по одной оси и
сначала проверять с `--dry-run`. Для RARS01 `End_link` ставится по переднему
краю GraspNet: `translation + depth * approach_axis`; `insertion_depth_m` должен
оставаться равным `0`.

### Проверка нового рычажного гриппера

В конфиге задана геометрия: радиус центрального рычага 37.5 мм, короткие
тяги 40 мм, каретки 30 мм, полезное раскрытие 100 мм и глубина 80 мм.
Сначала вывести расчёт без подключения робота:

```bash
.venv/bin/python scripts/check_gripper_linkage.py \
  --config config/jetson_orin_nano.yaml
```

Для проверки штангенциркулем вручную полностью закрыть губки при выключенных
моторах, поставить руку в home, подготовить аварийную остановку и явно добавить
`--execute`:

```bash
.venv/bin/python scripts/check_gripper_linkage.py \
  --config config/jetson_orin_nano.yaml --execute
```

После `START` скрипт сразу задаёт первое значение. Для следующих значений 40,
60, 80 и 100 мм ждёт Enter и печатает ошибку после ввода измеренного раскрытия. Моторы всегда
отключаются при выходе. Если направление движения неверное, немедленно
остановить тест и изменить только `robot.gripper.rars01.counterclockwise`.
Закрытая позиция берётся только из `robot.gripper.rars01.closed_position_rad`;
промежуточный feedback при включении не используется как ноль.

### Eye-in-hand калибровка камеры

После изменения крепления камеры или frame `End_link` выполните калибровку:

```bash
.venv/bin/python scripts/collect_handeye_eih.py \
  --config config/jetson_orin_nano.yaml --robot-backend rars01
```

На Jetson установленный OpenCV может не содержать `cv2.calibrateHandEye`.
Дополнительно устанавливать OpenCV не нужно: скрипт автоматически использует
совместимый NumPy-решатель. Результат сохраняется в
`config/calibration/orbbec_gemini2/hand_eye.npz`, а исходные измерения — в
`config/calibration/orbbec_gemini2/hand_eye_samples.npz`; прежний результат
перезаписывается только после успешного решения.
