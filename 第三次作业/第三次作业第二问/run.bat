@echo off
echo 开始 COLMAP 重建

:: 自动创建文件夹
mkdir data\colmap
mkdir data\colmap\sparse
mkdir data\colmap\dense

:: 1 特征提取
"E:\colmap\bin\colmap.exe" feature_extractor --database_path data/colmap/database.db --image_path data/images --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1

:: 2 特征匹配
"E:\colmap\bin\colmap.exe" exhaustive_matcher --database_path data/colmap/database.db

:: 3 稀疏重建
"E:\colmap\bin\colmap.exe" mapper --database_path data/colmap/database.db --image_path data/images --output_path data/colmap/sparse

:: 4 去畸变
"E:\colmap\bin\colmap.exe" image_undistorter --image_path data/images --input_path data/colmap/sparse/0 --output_path data/colmap/dense

:: 5 稠密重建
"E:\colmap\bin\colmap.exe" patch_match_stereo --workspace_path data/colmap/dense

:: 6 生成点云
"E:\colmap\bin\colmap.exe" stereo_fusion --workspace_path data/colmap/dense --output_path data/colmap/dense/fused.ply

echo 重建完成！
pause