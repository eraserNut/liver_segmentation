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

| <sub>Model</sub> | <sub>Dice</sub> | <sub>F1</sub> | <sub>Backbone</sub> |  <sub>use </br> pretrained </sub>|
|:---------------------------------:|:-:|:------------------------:|:--------------------:|
|<sub>Unet</sub>| 0.9494 | 0.9503 | <sub>resnet18</sub>|yes|
|<sub>Unet++</sub>| - | - | <sub>resnet18</sub>|yes|
|<sub>FPN</sub>| 0.9493 | 0.9491 | <sub>resnet18</sub>|yes|
|<sub>DAF</sub>| - | - | <sub>resnext101</sub>|yes|