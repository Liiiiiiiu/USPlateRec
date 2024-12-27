python export.py \
        --weights "/mnt/jx/car_plate_rec/output/360CC/crnn_fintune/2024-12-25-15-30/checkpoints/checkpoint_84_acc_0.8523.pth" \
        --save_path "/mnt/jx/car_plate_rec/onnx_jx/mid_88florida_val_1225.onnx" \
        --img_size 48 168 \
        --batch_size 1 \
        --dynamic  \
        --simplify 

