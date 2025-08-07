#!/bin/bash

torchrun --nproc_per_node=2 examples/torchvision/image_classification.py --world_size 2 --config configs/experiment/ilsvrc2012/ce-resnet18-batch128x2.yaml --run_log log/ilsvrc2012/ce/resnet18.txt
torchrun --nproc_per_node=2 examples/torchvision/image_classification.py --world_size 2 --config configs/experiment/ilsvrc2012/kd-resnet18_from_resnet34-batch128x2.yaml --run_log log/ilsvrc2012/kd/resnet18_from_resnet34.txt
