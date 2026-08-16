说明：把best.pt放到./model/里，使用环境名rdkx5_convert_clean

在虚拟机中执行：
    cd /home/li
    unzip "/home/li/下载/RDK_X5_YOLOv8_Conda_OneClick_v4.zip"
    mv RDK_X5_YOLOv8_Conda_OneClick_v4 rdk_x5_converter 
    cd /home/li/rdk_x5_converter
确认当前路径：
    pwd
应当显示：
    /home/li/rdk_x5_converter
将30张左右的原始图片放入：
    calibration_images/
执行：
    cd /home/li/rdk_x5_converter
    bash run_all.sh
成功后模型位于：
    output/best_bayese_640x640_nv12.bin