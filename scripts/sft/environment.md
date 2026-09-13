
```
# python --version

Python 3.10.17
```

```
# pip list

Package                       Version                   Editable project location
----------------------------- ------------------------- ----------------------------
accelerate                    1.10.0
aiohappyeyeballs              2.6.1
aiohttp                       3.12.15
aiosignal                     1.4.0
annotated-types               0.7.0
antlr4-python3-runtime        4.9.3
apex                          0.1.dev20250214+ascend
ascend-faultdiag              7.1rc1.post20260109175653
async-timeout                 5.0.1
attrs                         25.3.0
bitsandbytes-npu-beta         0.45.3
certifi                       2025.8.3
charset-normalizer            3.4.3
click                         8.2.1
codetiming                    1.4.0
datasets                      4.0.0
decorator                     5.2.1
dill                          0.3.8
einops                        0.8.1
exceptiongroup                1.3.0
filelock                      3.18.0
frozenlist                    1.7.0
fsspec                        2025.3.0
gpytorch                      1.14
greenlet                      3.2.4
hf-xet                        1.1.7
highspy                       1.9.0
huggingface-hub               0.34.4
idna                          3.10
iniconfig                     2.1.0
jaxtyping                     0.3.2
Jinja2                        3.1.6
joblib                        1.5.1
jsonschema                    4.25.0
jsonschema-specifications     2025.4.1
latex2sympy2                  1.5.4
linear-operator               0.6
MarkupSafe                    3.0.2
mindspeed                     0.12.1                    /MindSpeed-LLM/MindSpeed
mindspeed-llm                 2.3.0                     /MindSpeed-LLM/MindSpeed-LLM
mistral_common                1.8.3
mpmath                        1.3.0
msgpack                       1.1.1
multidict                     6.6.3
multiprocess                  0.70.16
networkx                      3.4.2
ninja                         1.11.1.4
numpy                         1.26.0
packaging                     25.0
pandas                        2.3.1
peft                          0.7.1
pillow                        12.1.0
pip                           23.0.1
pluggy                        1.6.0
ply                           3.11
propcache                     0.3.2
protobuf                      6.31.1
psutil                        7.0.0
PuLP                          3.0.0
pyarrow                       21.0.0
pybind11                      3.0.0
pycountry                     24.6.1
pydantic                      2.12.5
pydantic_core                 2.41.5
pydantic-extra-types          2.11.0
Pygments                      2.19.2
pytest                        8.4.1
pytest-mock                   3.15.1
python-dateutil               2.9.0.post0
pytz                          2025.2
PyYAML                        6.0.2
ray                           2.10.0
referencing                   0.36.2
regex                         2025.7.33
requests                      2.32.4
rpds-py                       0.27.0
safetensors                   0.6.2
scikit-learn                  1.7.1
scipy                         1.15.3
sentencepiece                 0.2.0
setuptools                    65.5.0
six                           1.17.0
SQLAlchemy                    2.0.42
sympy                         1.14.0
threadpoolctl                 3.6.0
tiktoken                      0.11.0
tokenizers                    0.22.2
tomli                         2.2.1
torch                         2.7.1
torch_npu                     2.7.1
tqdm                          4.67.1
transformers                  4.57.1
transformers-stream-generator 0.0.5
typing_extensions             4.14.1
typing-inspection             0.4.2
tzdata                        2025.2
urllib3                       2.5.0
wadler_lindig                 0.1.7
wheel                         0.45.1
word2number                   1.1
xxhash                        3.5.0
yarl                          1.20.1
```

```
node0 npu-smi info

+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.1                   Version: 25.5.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B1               | OK            | 97.4        50                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3425 / 65536         |
+===========================+===============+====================================================+
| 1     910B1               | OK            | 100.5       51                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3416 / 65536         |
+===========================+===============+====================================================+
| 2     910B1               | OK            | 96.3        50                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3415 / 65536         |
+===========================+===============+====================================================+
| 3     910B1               | OK            | 100.7       51                0    / 0             |
| 0                         | 0000:02:00.0  | 0           0    / 0          3414 / 65536         |
+===========================+===============+====================================================+
| 4     910B1               | OK            | 98.2        50                0    / 0             |
| 0                         | 0000:81:00.0  | 0           0    / 0          3416 / 65536         |
+===========================+===============+====================================================+
| 5     910B1               | OK            | 103.9       53                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          3417 / 65536         |
+===========================+===============+====================================================+
| 6     910B1               | OK            | 102.8       51                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          3416 / 65536         |
+===========================+===============+====================================================+
| 7     910B1               | OK            | 97.5        52                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          3416 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 2                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 3                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 4                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 6                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 7                                                            |
+===========================+===============+====================================================+
```

```
CANN version: cann_8.3rc3
```