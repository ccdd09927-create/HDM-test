# Euler Validation Checkpoint Selection

| PDE | Model | Role | Step | Val relL2 mean | Selected |
|---|---|---|---:|---:|---|
| compressible_ns | FNO | val_epoch_1000 | 1000 | 3.1214759349823 | no |
| compressible_ns | FNO | val_epoch_2000 | 2000 | 1.5628615617752075 | no |
| compressible_ns | FNO | val_epoch_3000 | 3000 | 2.1187822818756104 | no |
| compressible_ns | FNO | val_epoch_4000 | 4000 | 1.583549976348877 | no |
| compressible_ns | FNO | val_epoch_5000 | 5000 | 1.3116785287857056 | no |
| compressible_ns | FNO | val_epoch_6000 | 6000 | 1.5322091579437256 | no |
| compressible_ns | FNO | val_epoch_7000 | 7000 | 1.5144407749176025 | no |
| compressible_ns | FNO | val_epoch_8000 | 8000 | 0.6920818090438843 | no |
| compressible_ns | FNO | val_epoch_9000 | 9000 | 0.7127746343612671 | no |
| compressible_ns | FNO | val_epoch_10000 | 10000 | 0.5475017428398132 | yes |
| compressible_ns | KNO(head=1) | val_epoch_1000 | 1000 | 5.830753326416016 | no |
| compressible_ns | KNO(head=1) | val_epoch_2000 | 2000 | 3.2854909896850586 | no |
| compressible_ns | KNO(head=1) | val_epoch_3000 | 3000 | 3.0217785835266113 | no |
| compressible_ns | KNO(head=1) | val_epoch_4000 | 4000 | 1.4300053119659424 | no |
| compressible_ns | KNO(head=1) | val_epoch_5000 | 5000 | 1.459923267364502 | no |
| compressible_ns | KNO(head=1) | val_epoch_6000 | 6000 | 0.5561148524284363 | no |
| compressible_ns | KNO(head=1) | val_epoch_7000 | 7000 | 0.32287493348121643 | no |
| compressible_ns | KNO(head=1) | val_epoch_8000 | 8000 | 0.27401262521743774 | no |
| compressible_ns | KNO(head=1) | val_epoch_9000 | 9000 | 0.4595628082752228 | no |
| compressible_ns | KNO(head=1) | val_epoch_10000 | 10000 | 0.15495023131370544 | yes |
| compressible_ns | MHLKNO | val_epoch_1000 | 1000 | 3.8111214637756348 | no |
| compressible_ns | MHLKNO | val_epoch_2000 | 2000 | 2.716134548187256 | no |
| compressible_ns | MHLKNO | val_epoch_3000 | 3000 | 3.190227508544922 | no |
| compressible_ns | MHLKNO | val_epoch_4000 | 4000 | 1.9627711772918701 | no |
| compressible_ns | MHLKNO | val_epoch_5000 | 5000 | 1.1349396705627441 | no |
| compressible_ns | MHLKNO | val_epoch_6000 | 6000 | 0.3897649943828583 | no |
| compressible_ns | MHLKNO | val_epoch_7000 | 7000 | 0.4672338366508484 | no |
| compressible_ns | MHLKNO | val_epoch_8000 | 8000 | 0.39052683115005493 | no |
| compressible_ns | MHLKNO | val_epoch_9000 | 9000 | 0.43180984258651733 | no |
| compressible_ns | MHLKNO | val_epoch_10000 | 10000 | 0.2263362556695938 | yes |
| compressible_ns | MHLKNO_LINATTN | val_epoch_1000 | 1000 | 5.561036109924316 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_2000 | 2000 | 3.480621337890625 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_3000 | 3000 | 3.560006856918335 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_4000 | 4000 | 2.100337505340576 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_5000 | 5000 | 0.9680694341659546 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_6000 | 6000 | 0.32413536310195923 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_7000 | 7000 | 0.5640764236450195 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_8000 | 8000 | 0.26287493109703064 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_9000 | 9000 | 0.3210274279117584 | no |
| compressible_ns | MHLKNO_LINATTN | val_epoch_10000 | 10000 | 0.18773718178272247 | yes |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_1000 | 1000 | 5.555202007293701 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_2000 | 2000 | 3.4741334915161133 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_3000 | 3000 | 3.5683300495147705 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_4000 | 4000 | 2.0232481956481934 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_5000 | 5000 | 0.8915177583694458 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_6000 | 6000 | 0.32234615087509155 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_7000 | 7000 | 0.5332200527191162 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_8000 | 8000 | 0.2615495026111603 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_9000 | 9000 | 0.32080215215682983 | no |
| compressible_ns | MHLKNO_LINATTN_ablation | val_epoch_10000 | 10000 | 0.1868007928133011 | yes |

Legacy `ckpt_best_val.pth` files are treated as Tsit5-selected when they predate the Euler reselection and are not used as the default best-validation checkpoint.
