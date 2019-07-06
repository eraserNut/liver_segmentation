###Experiments records for FPN based segmentation method
|FPN version|dice|F1|note|
|:---:|:---:|:---:|:---:|
|FPN_baseline|0.8657|0.8656|<sub>the baseline method</sub>
|FPN_DS_V1|0.8663|0.8655|<sub>add raw DS module</sub>
|FPN_DS_V1|0.8671|0.8657|<sub>cross gt</sub>
|FPN_DS_V1|0.8688|0.8685|<sub>modify weight of fn,fp loss</sub>
|FPN_DS_V2|0.8688|0.8660|<sub>modify DS module, straightforward train mask</sub>
|FPN_DS_V2|0.8651|0.8624|<sub>cross gt</sub>
|FPN_DS_V3|0.8685|0.8677|<sub>fuse and union train mask, use union mask to each layer</sub>
|FPN_DS_V3|0.8692|0000|<sub>cross gt</sub>
|FPN_DS_V4|0.8646|0.8706|<sub>add DUpsampling module</sub>
|FPN_Dup|0000|0000|<sub>add DUpsampling module</sub>
|FPN_multi_task|0.8722|0.8736|<sub>add multi-task learning, just share backbone</sub>
|FPN_multi_task_V2|0000|0000|<sub>multi-task learning, share all</sub>
