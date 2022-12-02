# Contrastive Learning for Video Summarization

## Abstract

Query-based video summarization aims to provide customized video summaries based on a semantic query provided by a user.  [CLIP-IT](https://arxiv.org/pdf/2107.00650.pdf) pursues a solution that formulates this task as a per-frame binary classification problem. However, we believe such a problem formulation precludes architectures from applying appropriate levels of consideration to the temporal aspects of considered frames, given that keyframes are still-images at a single timestep. Furthermore, desirable video summaries should avoid unnecessarily frequent jump cuts between non-contiguous keyframes: rapidly cycling through highly diverse images may make viewers nauseated and unable to fully comprehend the video summary. Given the success of contrastive learning in domains where there exists a dearth of labeled data, [TCLR](https://arxiv.org/pdf/2101.07974.pdf) was able to demonstrate the effectiveness of contrastive learned video representations that account for the temporal characteristics unique to video data. We seek to leverage these learned representations to improve performance on the query-based video summarization task.



## Instructions
Run the following to train our TCLR network.
```
cd TCLR
python sumtclr_train_gen_all_step.py --run_id 'd-augmented_tvsum' --num_epochs 3 --num_dataloader_workers 4 --traintestlist ../data/splits/augmented_tvsum_80.txt --repeats 1 --batch_size=8 | tee d-augmented_tclr.log
```