from rknn.api import RKNN
rknn = RKNN(verbose=True)
print('--> Loading model')

ret = rknn.load_tensorflow(tf_pb='./frozen_actions.pb',inputs=['input'],outputs=['actions'],input_size_list=[[84,84,4]])

if ret !=0:
    print('Load failed!')
    exit(ret)
print('done')



print('--> Building model') 
ret = rknn.build(do_quantization=False)

if ret !=0:
    print('Build failed!')
    exit(ret)

print('done')

print('--> Export RKNN model')
ret = rknn.export_rknn('./final.rknn') 

if ret != 0:
    print('Export failed!')
    exit(ret)

print('done')

