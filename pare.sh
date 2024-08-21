module load cuda/11.1.1 gcc/7.5.0 ffmpeg/6.0
python demo/estimate_smpl.py \
    configs/pare/hrnet_w32_conv_pare_mix_cache.py \
    data/checkpoints/hrnet_w32_conv_pare_mosh.pth \
    --single_person_demo \
    --det_config demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --show_path vis_results/single_person_demo.mp4 \
    --output result \
    --input_path /gammascratch/xiyichen/bike/all_rgb \
    # --input_path  demo/resources/single_person_demo.mp4 \

