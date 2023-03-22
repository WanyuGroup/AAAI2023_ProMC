# This repo covers the implementation for our paper ProMC

Zhenzhong Wang, Lulu Cao, Wanyu Lin, Jiang Min, and Kaychen Tan. "Robust Graph Meta Learning via Manifold Calibration with Proxy Subgraphs," in the Proceedings of the Thirty-Seventh Conference on Association for the Advancement of Artificial Intelligence (AAAI), Washington DC, USA, Feburary 7-14, 2023.

Dependencies
-----

The script has been tested running under Python 3.8, with the following packages installed:

- `torch: 1.11.0   `
- `dgl: 0.8.2   `
- `scipy: 1.9.1 `
- `numpy: 1.21.6 `

In addition, CUDA 11.3 has been used in our project

Data Processing
-----
We use the attack methods provided in DeepRobust [code](https://github.com/DSE-MSU/DeepRobust) to process the data.


Run
-----

       python main.py


Cite Us
-----
```
@inproceedings{promcgraphnnaaai2023,
	title = {Robust Graph Meta Learning via Manifold Calibration with Proxy Subgraphs},
	author = {Wang, Zhenzhong and Cao, Lulu and Lin, Wanyu and Jiang, Min and Tan, Kaychen},
	booktitle = {Proceedings of the Thirty-Seventh Conference on Association for the Advancement of Artificial Intelligence (AAAI)},
	year = {2023}
}
```
