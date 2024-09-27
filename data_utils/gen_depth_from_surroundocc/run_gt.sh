export OptiX_INSTALL_DIR=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
export CUDA_HOME=/mnt/data/home/fuquan.jin/miniconda/envs/triro_optix
export TORCH_CUDA_ARCH_LIST="8.6"
echo ${CUDA_HOME}
echo ${OptiX_INSTALL_DIR}
for i in {10..19..10}; do
	start_seq=$i
	end_seq=$((i+10))
	echo "process ${start_seq} - ${end_seq}"
	#export CUDA_VISIBLE_DEVICES=$((i/10))
	#sleep 60
	nohup python3 generate_occupancy_nuscenes_gpu.py  --split train --save_path /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/nuscenes_occ_gt_2 --dataroot /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/nuscenes_fqj --nusc_val_list /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/SurroundOcc-main/tools/generate_occupancy_nuscenes/nuscenes_val_list.txt --label_mapping /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/SurroundOcc-main/tools/generate_occupancy_nuscenes/nuscenes.yaml --config_path /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/SurroundOcc-main/tools/generate_occupancy_nuscenes/config.yaml --start ${start_seq} --end ${end_seq} | tee "process_${start_seq}-${end_seq}".log & 
done
