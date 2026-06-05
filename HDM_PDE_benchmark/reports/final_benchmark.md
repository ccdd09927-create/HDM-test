# Final benchmark

| PDE | Model | Epochs | Selected ckpt | NFE | Sampler | Test samples | Physical relL2 mean | Notes |
|---|---|---:|---|---:|---|---:|---:|---|
| compressible_ns | FNO | 10000 | best_val_euler_physical_l2 (10000) | 200 | euler | 64 | 0.513514518737793 | evaluated; selected=best_val_euler_physical_l2; best_validation_physical_l2_checkpoint_reselected_with_euler; ignored_for_selection_tsit5_selected_before_euler_reselection; euler_val_candidates=10 |
| compressible_ns | KNO(head=1) | 10000 | best_val_euler_physical_l2 (10000) | 200 | euler | 64 | 0.13372084498405457 | evaluated; selected=best_val_euler_physical_l2; unconditional_nonfno_epoch_sweep_euler_best_val_remained_best; ignored_for_selection_tsit5_selected_before_euler_reselection; euler_val_candidates=10 |
| compressible_ns | MHLKNO | 10000 | best_val_euler_physical_l2 (10000) | 200 | euler | 64 | 0.1454758644104004 | evaluated; selected=best_val_euler_physical_l2; unconditional_nonfno_epoch_sweep_euler_best_val_remained_best; ignored_for_selection_tsit5_selected_before_euler_reselection; euler_val_candidates=10 |
| compressible_ns | MHLKNO_LINATTN | 10000 | best_val_euler_physical_l2 (10000) | 200 | euler | 64 | 0.12775090336799622 | evaluated; selected=best_val_euler_physical_l2; unconditional_nonfno_epoch_sweep_euler_best_val_remained_best; ignored_for_selection_tsit5_selected_before_euler_reselection; euler_val_candidates=10 |
| compressible_ns | MHLKNO_LINATTN_ablation | 10000 | best_val_euler_physical_l2 (10000) | 200 | euler | 64 | 0.12685824930667877 | evaluated; selected=best_val_euler_physical_l2; unconditional_nonfno_epoch_sweep_euler_best_val_remained_best; ignored_for_selection_tsit5_selected_before_euler_reselection; euler_val_candidates=10 |
