cam_list=(CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT)
data_path=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/nuscenes/samples
out_base_path=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin
source /mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/setup.bash
engine=/mnt/nas/algorithm/xianda.guo/intern/fuquan.jin/code/Depth-Anything/6fda5199f0043e91e1c6117df00313db57c630d8ed0306070b43da1ecfffbd9d_sim.engine

for cam_index in ${!cam_list[@]}
do
	export CUDA_VISIBLE_DEVICES=${cam_index}
	echo ${CUDA_VISIBLE_DEVICES}
	img_path=${data_path}/${cam_list[$cam_index]}
	out_path=${out_base_path}/${cam_list[$cam_index]}
	if [ -d ${out_path} ]
	then
		echo "${out_path} exists"
	else
		mkdir ${out_path}
		echo "${out_path} created"
	fi
	#echo ${img_path} ${out_path} && sleep 10 && echo "done ${img_path}" &
        nohup python python/trt_infer.py --img ${img_path} --outdir ${out_path} --engine  ${engine} --grayscale &> ${out_path}/predict.log &
	#echo $(ls $img_path | wc -l)
done
