# ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning
PyTorch implementation for [ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning](https://arxiv.org/abs/2110.02027) accepted by ICML 2022.
## Requirements
* Python 3.7.4
* PyTorch 1.7.0
* torch_geometric 1.5.0
* tqdm
## Training & Evaluation
ProGCL-weight:
```
python train.py --device cuda:0 --dataset Amazon-Computers --param local:amazon-computers.json --mode weight
```
ProGCL-mix:
```
python train.py --device cuda:0 --dataset Amazon-Computers --param local:amazon-computers.json --mode mix
```
## Citation
```
@inproceedings{xia2022progcl,
  title={ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning},
  author={Xia, Jun and Wu, Lirong and Wang, Ge and Li, Stan Z.},
  booktitle={International conference on machine learning},
  year={2022},
  organization={PMLR}
}
```
## Useful resources for Pretrained Graphs Neural Networks
* The first comprehensive survey on this topic: [A Survey of Pretraining on Graphs: Taxonomy, Methods, and Applications](https://arxiv.org/abs/2202.07893v1)
* [A curated list of must-read papers, open-source pretrained models and pretraining datasets.](https://github.com/junxia97/awesome-pretrain-on-graphs)
