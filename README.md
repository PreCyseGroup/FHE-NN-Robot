# Encrypted Cloud-Based Neural Network Controller for Differential-Drive Robots

This repository accompanies the paper **"Encrypted Cloud-Based Neural Network Controller for Differential-Drive Robots"**.

Cloud-based robot control can provide access to powerful remote computing resources, but it also creates privacy concerns because sensitive robot states, reference commands, and control outputs may be exposed to the cloud. This work investigates how homomorphic encryption can be used to protect this information while still allowing a remote server to perform neural-network-based control.

The proposed approach uses a compact neural network with a quadratic polynomial activation function to approximate a classical Kanayama tracking controller. During online operation, the robot computes its tracking features locally, encrypts them using the CKKS homomorphic encryption scheme, and sends them to the server. The server evaluates the neural network directly on the encrypted data without decrypting it, and the encrypted control output is returned to the robot for local decryption and actuation.

The framework is implemented in Webots using a client-server architecture and is evaluated by comparing three controller modes:

- **Kanayama controller**: the classical analytic tracking controller.
- **Plaintext neural network**: the trained neural network evaluated without encryption.
- **CKKS neural network**: the trained neural network evaluated directly on encrypted data.

The experiments examine both trajectory-tracking performance and the computational cost of encrypted inference.

<img width="1947" height="1087" alt="architecture_new" src="https://github.com/user-attachments/assets/d14f8283-6bbb-4074-8ceb-e24a11e12878" />

*System architecture of the proposed encrypted cloud-based neural controller.*

This repository provides the Webots simulation, dataset generation, neural-network training, client-server inference, and result-analysis code used in the study.

---

## Repository contents

| File / directory | Purpose |
|---|---|
| `data_collector.py` | Webots controller used to generate the training dataset with the Kanayama controller. |
| `train.py` | Trains the 5-64-2 neural network and exports `model_export.json`. |
| `client.py` | Webots evaluation client. It supports Kanayama, plaintext NN, and CKKS NN modes and saves the run data. |
| `server.py` | Inference server for the three controller modes. |
| `requirements.txt` | Python dependencies. |
| `Webots/world/world.wbt` | Webots world used for data collection and evaluation. |
| `Webots/controllers/robot_controller/robot_controller.py` | The controller file actually executed by Webots. It is overwritten with either `data_collector.py` or `client.py` depending on the experiment stage. |
| `scripts/results.py` | Loads the three experiment runs, prints tracking/timing metrics, and displays the comparison plots. |

### Generated files

| File | Created by | Used by |
|---|---|---|
| `Webots/webots_dataset.npz` | `data_collector.py` in Webots | copied next to `train.py` |
| `webots_dataset.npz` | copied from `Webots/` | `train.py` |
| `model_export.json` | `train.py` | `server.py` |
| `Webots/model_export.json` | copy of the trained model | Webots evaluation client |
| `Webots/run_data_webots_kanayama.npz` | Webots client | `scripts/results.py` |
| `Webots/run_data_webots_plaintext.npz` | Webots client | `scripts/results.py` |
| `Webots/run_data_webots_ckks.npz` | Webots client | `scripts/results.py` |
| `server_processing_times_*.npz` | `server.py` | `scripts/results.py` |

The workflow tested for this repository is **Windows**. The commands below therefore use Windows Command Prompt syntax.

---

## Experiment configuration

### Robot and controller

| Parameter | Value |
|---|---:|
| Webots simulation step | 10 ms |
| Control update period | 100 ms |
| Wheel radius | 0.04445 m |
| Wheelbase | 0.393 m |
| Wheel-speed limit | +/-10 rad/s |
| Kanayama gains | `Kx=2.0`, `Ky=9.0`, `Ktheta=6.0` |

The five neural-network inputs are:

```text
[e_x, e_y, e_theta, v_ref, omega_ref]
```

The two outputs are the right and left wheel commands:

```text
[wr, wl]
```

### Training data

The training dataset uses eight reference trajectories:

```text
lemniscate
circle
rounded_square
spiral
rose
clover
hypotrochoid
random_spline
```

The Lissajous trajectory is **not** used for training. It is reserved for the final evaluation.

Dataset settings:

- 16 evenly distributed rollout starting points per training trajectory
- 32 initial tracking-error states per starting point
- 40 control samples per rollout (`4.0 s` at `Ts = 0.1 s`)
- trajectory seed: `42`
- scrambled Sobol perturbation seed: `43`
- initial `e_x` range: `[-0.30, 0.30] m`
- initial `e_y` range: `[-0.30, 0.30] m`
- initial `e_theta` range: `[-0.70, 0.70] rad`
- one perturbation state is explicitly set to the nominal zero-error state `[0, 0, 0]`
- 512 rollouts per training trajectory (`16 x 32`)
- 20,480 feature-target pairs per trajectory (`16 x 32 x 40`)
- total dataset size: **163,840 feature-target pairs**

### Neural network

- architecture: `5 -> 64 -> 2`
- activation: `p(z) = z + 0.125 z^2`
- optimizer: AdamW
- epochs: `16`
- batch size: `1024`
- learning rate: `3e-3`
- weight decay: `1e-5`

### CKKS

- polynomial modulus degree: `8192`
- coefficient modulus bits: `[40, 30, 30, 30, 30, 40]`
- global scale: `2^30`
- Galois keys: enabled
- relinearization keys: enabled

### Evaluation

- unseen trajectory: 50 s Lissajous path
- measured control requests: 501 (`t = 0.0, 0.1, ..., 50.0 s`)
- two additional warm-up requests are executed before the measured run
- remote inference runs in a background thread; the most recently available wheel command is held until the next response arrives

For the paper experiment, the Webots client and the inference server ran on the same workstation and communicated through the loopback interface (`127.0.0.1`).

The paper test workstation used an Intel Core i9-14900K processor, an NVIDIA RTX 2000 Ada Generation GPU with 16 GB of GPU memory, and 32 GB of DDR5 RAM.

---

# Reproducing the experiment

## 1. Install the dependencies

Install the packages from the repository root:

```bat
pip install -r requirements.txt
```

Check the main dependencies:

```bat
python -c "import numpy, scipy, torch; print('NumPy/SciPy/PyTorch OK')"
```

For CKKS, also check TenSEAL:

```bat
python -c "import tenseal; print('TenSEAL OK')"
```

The Webots Python controller must use a Python environment in which the required packages are installed.

---

## 2. Generate the Webots training dataset

Webots is configured to execute:

```text
Webots/controllers/robot_controller/robot_controller.py
```

For data collection, overwrite that file with `data_collector.py`:

```bat
copy /Y data_collector.py Webots\controllers\robot_controller\robot_controller.py
```

Open the following world in Webots:

```text
Webots/world/world.wbt
```

Reset the simulation and run it in **fast mode (`>>`)**. The collector prints occasional progress messages while it processes the eight trajectories. For each trajectory it uses 16 starting points and 32 initial tracking-error states, producing 512 short recovery rollouts of 40 control samples each.

When collection is complete, the following file is created:

```text
Webots/webots_dataset.npz
```

The expected number of samples is:

```text
163840
```

Copy the dataset to the repository root so that it is next to `train.py`:

```bat
copy /Y Webots\webots_dataset.npz webots_dataset.npz
```

At this point the repository root should contain:

```text
train.py
webots_dataset.npz
```

---

## 3. Train and export the neural network

Run:

```bat
python train.py
```

The script loads `webots_dataset.npz`, calculates the normalization values, trains the neural network, and creates:

```text
model_export.json
```

Keep this file in the repository root because `server.py` reads it there by default.

The Webots client also needs the same model file for the normalization values. Copy it into the Webots folder:

```bat
copy /Y model_export.json Webots\model_export.json
```

You should now have:

```text
model_export.json
Webots/model_export.json
```

These two files must be identical.

---

## 4. Switch Webots from data collection to evaluation

Replace the active Webots controller with `client.py`:

```bat
copy /Y client.py Webots\controllers\robot_controller\robot_controller.py
```

Reset Webots after replacing the controller file.

The evaluation client draws the Lissajous reference path directly in the Webots world, so no separate trajectory object needs to be added to `world.wbt`.

---

## 5. Start the inference server

Open a terminal in the repository root and run:

```bat
python server.py
```

Keep this terminal open while Webots is running.

Default connection:

```text
server address: 127.0.0.1
port: 50007
```

The server runs in `auto` mode by default and accepts the mode requested by the client.

For plaintext and CKKS modes, make sure `model_export.json` is next to `server.py`.

---

## 6. Run the three controller modes

The evaluation client supports:

```text
kanayama
plaintext
ckks
```

The active Webots controller contains a line equivalent to:

```python
MODE = os.environ.get("AMR_MODE", "ckks").lower()
```

The most direct method on Windows is to change the default string before each run:

```python
"kanayama"
```

then:

```python
"plaintext"
```

then:

```python
"ckks"
```

Reset Webots before every new run.

For each mode:

1. make sure `server.py` is running;
2. select the desired client mode;
3. reset the Webots simulation;
4. start the simulation;
5. wait for the complete 50 s run;
6. wait for the client to save the result file;
7. wait for the server to detect the disconnect and save its processing-time file.

The Webots client produces:

```text
Webots/run_data_webots_kanayama.npz
Webots/run_data_webots_plaintext.npz
Webots/run_data_webots_ckks.npz
```

The server produces:

```text
server_processing_times_kanayama.npz
server_processing_times_plaintext.npz
server_processing_times_ckks.npz
```

The server timing files include the two startup warm-up requests. `scripts/results.py` excludes them when calculating the measured results.

---

## 7. Collect the result files

Copy the three Webots run files to the repository root:

```bat
copy /Y Webots\run_data_webots_kanayama.npz run_data_webots_kanayama.npz
copy /Y Webots\run_data_webots_plaintext.npz run_data_webots_plaintext.npz
copy /Y Webots\run_data_webots_ckks.npz run_data_webots_ckks.npz
```

Before running the result script, the repository root should contain:

```text
run_data_webots_kanayama.npz
run_data_webots_plaintext.npz
run_data_webots_ckks.npz

server_processing_times_kanayama.npz
server_processing_times_plaintext.npz
server_processing_times_ckks.npz
```

---

## 8. Display the results

Run:

```bat
python scripts/results.py
```

The script:

- prints RMSE, mean position error, and maximum position error for the three controllers;
- prints the server processing and client-observed round-trip timing summaries;
- prints the CKKS timing breakdown;
- displays the trajectory comparison;
- displays the tracking-error comparison;
- displays the right- and left-wheel speed comparison.

The script only displays the figures; it does not save image files.

Timing values depend on the hardware, operating-system scheduling, and machine load, so they are not expected to exactly match the values obtained on the paper test machine.

---

## File flow

```text
data_collector.py
    |
    v
Webots/controllers/robot_controller/robot_controller.py
    |
    v
Webots/webots_dataset.npz
    |
    | copy to repository root
    v
webots_dataset.npz
    |
    v
train.py
    |
    v
model_export.json
    | \
    |  \\ copy
    |   v
    |  Webots/model_export.json
    |
    v
server.py

client.py
    |
    v
Webots/controllers/robot_controller/robot_controller.py
    |
    +--> run_data_webots_kanayama.npz
    +--> run_data_webots_plaintext.npz
    +--> run_data_webots_ckks.npz

server.py
    |
    +--> server_processing_times_kanayama.npz
    +--> server_processing_times_plaintext.npz
    +--> server_processing_times_ckks.npz

all six result files
    |
    v
scripts/results.py
```

---
