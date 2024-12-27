python export.py \
        --weights "/data_249/data2/jx/car_plate_train/car_plate_rec/output/360CC/lpr/2024-11-23-18-10/checkpoints/checkpoint_60_acc_0.8864.pth" \
        --save_path "/data_249/data2/jx/car_plate_train/car_plate_rec/onnx_jx/LPR_88_1025.onnx" \
        --img_size 24 94 \
        --batch_size 1 \
        --dynamic  \
        --simplify 

