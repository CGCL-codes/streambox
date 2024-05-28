import os

import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime
import sys
import json
from gluoncv import model_zoo, data, utils
from tvm.contrib.download import download_testdata

#####################################################
#
# This is an example of how to generate source code
# and schedule json from tvm.
#
# python3 vgg.py vgg.cu vgg.schedule.json vgg.graph.json vgg.params
#####################################################


if len(sys.argv) != 5:
    print("Usage: device_source_file_name raw_schedule_file graph_json_file param_file")
    exit(0)

file_path_dir = os.path.dirname(os.path.abspath(__file__)) + '/vgg'
if not os.path.exists(file_path_dir):
    os.mkdir(file_path_dir)
device_source_file = open("vgg/"+sys.argv[1], "w")  # cu
raw_schedule_file = open("vgg/"+sys.argv[2], "w")  # json
graph_json_file = open("vgg/"+sys.argv[3], "w")  # json
param_file = open("vgg/"+sys.argv[4], "w+b")  # params


print("begin build...")
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target, params=params)
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
print("get model...")
mod, params = relay.testing.vgg.get_workload(
    batch_size=1, image_shape=image_shape, num_layers=16 # vgg16
)

# mod, params = relay.testing.vgg.get_workload(
#     vgg_size=169, batch_size=batch_size, image_shape=image_shape
# )
# mod, params = relay.testing.mobilenet.get_workload(
#     batch_size=batch_size, image_shape=image_shape
# )

opt_level = 3
target = tvm.target.cuda()
# target = "cuda"
    
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

graph_json = lib.graph_json
params = lib.get_params()
ctx = tvm.gpu()
# ctx = tvm.device(target, 0)
module = graph_runtime.GraphModule(lib["default"](ctx))

tvm_input = np.ones(data_shape).astype("float32")
tvm_input = tvm_input * 10
# tvm_input = tvm.nd.array(x.asnumpy(), device=ctx)
module.set_input("data", tvm_input)

module.run()

device_source_file.write(lib.get_lib().imported_modules[0].get_source())
device_source_file.close()

graph_json_file.write(lib.get_json())
graph_json_file.close()

raw_schedule_file.write(module.module["get_schedule_json"]())
raw_schedule_file.close()


def dump_params(params, f):
    import array
    magic = bytes("TVM_MODEL_PARAMS\0", "ascii")
    f.write(magic)
    f.write(array.array('Q',[len(params)]).tobytes())
    for k in params.keys():
        # print(k)
        param = array.array('f', params[k].asnumpy().flatten().tolist())
        f.write(bytes(k, "ascii"))
        f.write(bytes("\0", "ascii"))
        f.write(array.array('Q',[len(param)]).tobytes())
        f.write(param.tobytes())

dump_params(params, param_file)
param_file.close()

# class_IDs, scores, bounding_boxs = module.get_output(0), module.get_output(1), module.get_output(2)
out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
print(out.flatten()[0:10])
# for i in range(10):
#     print(i + ":" + class_IDs[i] + scores[i] + bounding_boxs[i])