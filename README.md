# P2PNet (ICCV2021 Oral Presentation)
Follow the fork for the original stuff.
## Running in Google Cloud
¡¡¡¡WIP!!!!
The following implmentaion is thought to be used in Google Cloud. First, intal git and the run the .sh script
The ```rub_pred.py``` runs the inference pipeline over a dataset.

## Training

The network can be trained using the `train.py` script. For training on SHTechPartA, use

```
CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```
By default, a periodic evaluation will be conducted on the validation set.

## Testing

A trained model (with an MAE of **51.96**) on SHTechPartA is available at "./weights", run the following commands to launch a visualization demo:

```
CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./logs/
```