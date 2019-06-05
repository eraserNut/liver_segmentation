# Liver segmentation models

### Table of content
1. [Architectures](#architectures)
2. [Experiment results](#results)
3. ...

### Architectures <a name="architectures"></a>
- Unet
- Unet++
- FPN
- DAF

### Experiment results <a name="results"></a>

| <sub>Model</sub> | <sub>Dice</sub> | <sub>F1</sub> | <sub>Backbone</sub> | <sub>Batch size</sub>| <sub>Use pretrained </sub>|
|:-----------------------------:|:----:|:---------------------:|:--------------------:|:--------------------:|:--------------------:|
|<sub>Unet</sub>| 0.9494 | 0.9503 | <sub>Resnet-18</sub>|4|yes|
|<sub>Unet++</sub>| - | - | <sub>Resnet-18</sub>|2|yes|
|<sub>FPN</sub>| 0.9493 | 0.9491 | <sub>Resnet-18</sub>|4|yes|
|<sub>DAF</sub>| - | - | <sub>ResNext-101</sub>|4|yes|