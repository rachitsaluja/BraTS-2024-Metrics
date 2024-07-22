# BraTS-2024-Metrics

_Please note that this code is not the fully-optimized version yet. README will be modified more in the coming weeks._

This is an extension of this code repository - [BraTS 2023 Performance Metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics).

To use the code - 

```python
from metrics_{challenge} import *
results_df, _ = get_LesionWiseResults(pred_file, 
                      gt_file, 
                      challenge_name, 
                      output=None)
```

For the challenge name argument, the values are as follows - 

```
1. BraTS-GLI
2. BraTS-MEN-RT
3. BraTS-SSA
4. BraTS-PED
5. BraTS-MET
```

