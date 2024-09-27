export OptiX_INSTALL_DIR=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
export CUDA_HOME=/mnt/data/home/fuquan.jin/miniconda/envs/triro_optix
echo ${CUDA_HOME}
echo ${OptiX_INSTALL_DIR}
for i in {200..250..10}; do
	start_seq=$i
	end_seq=$((i+10))
	echo "process ${start_seq} - ${end_seq}"
	#export CUDA_VISIBLE_DEVICES=$((i/10-30))
	echo ${CUDA_VISIBLE_DEVICES}
	sleep 60
	nohup python3 generate_occupancy_nuscenes.py  --split train --save_path /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/nuscenes_occ_gt --dataroot /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/nuscenes_fqj --nusc_val_list /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/SurroundOcc-main/tools/generate_occupancy_nuscenes/nuscenes_val_list.txt --label_mapping /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/SurroundOcc-main/tools/generate_occupancy_nuscenes/nuscenes.yaml --config_path /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/SurroundOcc-main/tools/generate_occupancy_nuscenes/config.yaml --start ${start_seq} --end ${end_seq} | tee "process_${start_seq} - ${end_seq}".log & 
done
