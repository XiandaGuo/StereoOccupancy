import argparse
import os
import time
import cv2
import json
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from depth_anything.util.transform import load_image
from tqdm import tqdm

def read_filenames(img_path: str):
    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    return filenames

def predict(input_image, context):
    input_shape = context.get_tensor_shape('input')
    output_shape = context.get_tensor_shape('output')
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    # Copy the input image to the pagelocked memory
    np.copyto(h_input, input_image.ravel())
    
    # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    depth = h_output
    return depth, output_shape

def postprocess(depth, output_shape, orig_w, orig_h, colored : bool):
    # Process the depth output
    depth = np.reshape(depth, output_shape[2:])
    min_dpth = depth.min()
    max_dpth = depth.max()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    if colored:
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    depth = cv2.resize(depth, (orig_w, orig_h))
    return depth, min_dpth, max_dpth

def run(args):
    print(f"cuda visible divices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    # read filenames
    filenames = read_filenames(args.img)
    min_max_dpths = {}

    # Create logger and load the TensorRT engine
    start = time.perf_counter()
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    end = time.perf_counter()
    print(f"time of deserialize: {(end - start) * 1000}ms")

    start = time.perf_counter()
    context = engine.create_execution_context()
    end = time.perf_counter()
    print(f"time of creation of cuda device: {(end - start) * 1000}ms")
    
    colored = not args.grayscale

    print(f"colored :{colored}")
    for filename in tqdm(filenames):
        start = time.perf_counter()

        input_image, (orig_h, orig_w) = load_image(filename)
        depth, output_shape = predict(input_image, context)
        depth, min_dpth, max_dpth = postprocess(depth, output_shape, orig_w, orig_h, colored)
        print(min_dpth, max_dpth)

        # Save the depth map
        img_name = os.path.basename(filename)
        min_max_dpths[img_name] = {}
        min_max_dpths[img_name]['min_dpth'] = float(min_dpth)
        min_max_dpths[img_name]['max_dpth'] = float(max_dpth)
        dpth_name = f'{args.outdir}/{img_name[:img_name.rfind(".")]}_depth.png'
        cv2.imwrite(dpth_name, depth)

        end = time.perf_counter()
        print(f"time of {filename}: {(end-start) * 1000}ms")

    json_name = f'{args.outdir}/dpth.json'
    print(min_max_dpths)
    with open(json_name, 'w') as f:
        json.dump(min_max_dpths, f)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    
    args = parser.parse_args()
    print(args)
    run(args)
