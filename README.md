# Replication of Hierarchical Video-Moment Retrieval and Step-Captioning
My implementation of the code for the joint baseline model for the 4 hierarchical tasks for HiREST[[1]](#1), can be found in [main.ipynb](https://github.com/khushipatni-psu/CSE597_Project/blob/main/main.ipynb).

## Install Packages

```python
# Requires python 3.10.12
# Requires torch<1.13.0
# You need this only for step captioning evaluation (evaluate.py)
pip install allennlp_models

pip install -r requirements.txt
python -c "import language_evaluation; language_evaluation.download('coco')"
```

## Download Data

Download feature files from hugging face library extract them into the ./data/ directory.
Afterwards the ./data/ directory should look like:
```python
data/
    ASR/
    ASR_feats_all-MiniLM-L6-v2/
    eva_clip_features/
    eva_clip_features_32_frame/
    evaluation/
    splits/
```

You also need to download Clip4Caption and EVA-CLIP weights, and extract them into the ./pretrained_weights/ directory. 

## Run Training

```python
bash scripts/run.sh --train
```

## Inference & Evaluation

### Video Retrieval

```python
# Inference
python inference_video_retrieval.py \
    --data_dir './data/splits' \
    --video_feature_dir './data/eva_clip_features_32_frame' \
    --optim adamw \
    --n_model_frames 20 \
    --num_workers 4 \
    --eval_batch_size 10 \
    --device 'cuda' \
    --video_retrieval_model 'clip_g' \
    --run_name clip_g_VR_20frames_avgpool

# Evaluation
python evaluate.py \
    --task video_retrieval \
    --pred_data VR_results/clip_g_VR_20frames_avgpool.json
```

### Moment Retrieval / Moment Segmentation / Step Captioning

```python
# Inference
bash scripts/run.sh

# Evaluation
bash scripts/score.sh
```

# References
<a id="1">[1]</a> 
Abhay Zala and Jaemin Cho and Satwik Kottur and Xilun Chen and Barlas Oğuz and Yashar Mehdad and Mohit Bansal. 
Hierarchical Video-Moment Retrieval and Step-Captioning. 
CVPR, 2023.
