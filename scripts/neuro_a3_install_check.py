#!/usr/bin/env python3
"""A3 installation check: SANA-FE's bundled Loihi DVS-gesture example must reproduce its
documented output (tutorial/tutorial_5_dvs.ipynb at release v2.2.9: 365,277 neurons fired in
1,000 steps with the layer-to-core mapping [1, 4, 16, 16, 4, 1]). The network construction and
mapping below are the tutorial's cells, unchanged. Runs in .venv-neuro.

Output: results/neuro/a3_sanafe/install_check.json
"""
import json
import time
from pathlib import Path

import numpy as np
import sanafe
import sanafe.layers
import sanafe.tutorial

EXPECTED_FIRED = 365277
OUT = Path(__file__).resolve().parents[1] / "results/neuro/a3_sanafe/install_check.json"

arch = sanafe.load_loihi()
snn = sanafe.Network()
with sanafe.tutorial.get_dvs_data() as weight_file:
    data = np.load(weight_file)
    snn_attributes = {key: data[key] for key in data.files}
thresholds = snn_attributes["thresholds"]
biases = snn_attributes["inputs"]
layer0 = sanafe.layers.Input2D(snn, 32, 32, threshold=thresholds[0])
layer1 = sanafe.layers.Conv2D(snn, layer0, snn_attributes["conv1"], stride_width=2, stride_height=2, threshold=thresholds[1])
layer2 = sanafe.layers.Conv2D(snn, layer1, snn_attributes["conv2"], threshold=thresholds[2])
layer3 = sanafe.layers.Conv2D(snn, layer2, snn_attributes["conv3"], threshold=thresholds[3])
layer4 = sanafe.layers.Conv2D(snn, layer3, snn_attributes["conv4"], threshold=thresholds[4])
layer5 = sanafe.layers.Dense(snn, layer4, 11, snn_attributes["dense1"], threshold=thresholds[5])
for n, b in zip(layer0, biases):
    n.set_attributes(model_attributes={"bias": b})
layer_mapped_core_counts = [1, 4, 16, 16, 4, 1]
total_cores_mapped = 0


def map_layer_to_cores(layer, cores, core_count):
    global total_cores_mapped
    total_neurons = len(layer)
    neurons_per_core = total_neurons // core_count
    for idx in range(core_count):
        first_nid = idx * neurons_per_core
        is_last = (idx == (core_count - 1))
        if is_last:
            neurons_to_map_to_core = layer[first_nid:]
        else:
            last_nid = (idx + 1) * neurons_per_core
            neurons_to_map_to_core = layer[first_nid:last_nid]
        for neuron in neurons_to_map_to_core:
            neuron.map_to_core(cores[total_cores_mapped])
        total_cores_mapped += 1


for n in layer0:
    n.map_to_core(arch.tiles[0].cores[0])
cores = arch.cores()
for layer, c in zip((layer0, layer1, layer2, layer3, layer4, layer5), layer_mapped_core_counts):
    map_layer_to_cores(layer, cores, c)

t0 = time.time()
chip = sanafe.SpikingChip(arch)
chip.load(snn)
results = chip.sim(1000)
res = dict(expected_neurons_fired=EXPECTED_FIRED, neurons_fired=int(results["neurons_fired"]),
           passed=int(results["neurons_fired"]) == EXPECTED_FIRED, energy_total_J=float(results["energy"]["total"]),
           sim_time_s=float(results["sim_time"]), wall_seconds=round(time.time() - t0, 1),
           sanafe_version="2.2.9", reference="SANA-FE v2.2.9 tutorial/tutorial_5_dvs.ipynb, cell 6")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(res)
if not res["passed"]:
    raise SystemExit("SANA-FE does not reproduce its documented output")
