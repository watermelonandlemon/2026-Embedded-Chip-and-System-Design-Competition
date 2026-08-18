#代码功能： 此文件夹内程序，用于在不开启web节点、miniros等其他程序的情况下，单独运行视频检测程序
            best_bayese_640x640_nv12.bin  适合于BPU部署的模型文件
            camera.py  使用模型文件并读取摄像头视频流的python程序
            classes.names  模型文件中检测物体的类别(有先后顺序)
            readme.md  此文件夹内所有程序的说明文件
            run_camera.sh  脚本文件，运行可自动执行camera.py程序
#运行方式： ./run_camera.sh
#停止方式： q(退出)  s(保存)
#注意事项： 请在RDK本地运行，不要再Mobaxterm中运行，因为视频流的网络传输延迟会很大。