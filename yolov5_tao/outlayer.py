import tensorrt as trt





engine_path = '/home/ray/Documents/deepstream_tao_apps/models/yolov5/yolov5s.onnx_b1_gpu0_fp16.engine'

with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

for i in range(engine.num_bindings):
    binding_name = engine.get_binding_name(i)
    binding_shape = engine.get_binding_shape(i)
    if engine.binding_is_input(i):
        print(f"Binding {i} ({binding_name}): Input shape {binding_shape}")
    else:
        print(f"Binding {i} ({binding_name}): Output shape {binding_shape}")

