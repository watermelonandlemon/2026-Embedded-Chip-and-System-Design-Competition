说明：GPT已经把best.pt转换为BPU友好结构的best_yolov8s_bpu.onnx，放在./model/中

解决Anaconda安装后找不到的问题：
    /root/anaconda3/bin/conda init
    source ~/.bashrc

解压后进入目录：
    cd best_yolov8s_rdk_x5_bpu_package
使用 Conda 创建转换环境：
    conda create -n rdk_x5_mapper python=3.10 -y
    conda activate rdk_x5_mapper
    pip install -r conversion/requirements-convert.txt
    chmod +x conversion/convert_to_bin.sh
    ./conversion/convert_to_bin.sh
转换成功后生成：
    model/best_yolov8s_bpu_bayese_640x640_nv12.bin

把整个目录复制到开发板，然后执行：
    cd best_yolov8s_rdk_x5_bpu_package
    chmod +x runtime/run_image.sh
    ./runtime/run_image.sh \
        cal_images/170.jpg \
        result.jpg  
或者：
    python3 runtime/infer_image.py \
        --model model/best_yolov8s_bpu_bayese_640x640_nv12.bin \
        --image cal_images/170.jpg \
        --output result.jpg \
        --score 0.25 \
        --nms 0.70 \
        --bpu-cores 0