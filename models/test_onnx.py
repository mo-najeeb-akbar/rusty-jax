import sys
import onnxruntime as ort
import onnx
import numpy as np

def verify_onnx_fusion(onnx_path):
    """
    Verify that BatchNorm fusion worked in the exported ONNX model
    """
    print("=" * 50)
    print("ONNX BATCHNORM FUSION VERIFICATION")
    print("=" * 50)
    
    # Load ONNX model
    try:
        model = onnx.load(onnx_path)
        session = ort.InferenceSession(onnx_path)
        print(f"✅ Successfully loaded ONNX model: {onnx_path}")
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return
    
    # Count BatchNorm nodes in the graph
    batchnorm_nodes = []
    total_nodes = len(model.graph.node)
    
    for node in model.graph.node:
        if 'BatchNormalization' in node.op_type:
            batchnorm_nodes.append(node.name)
    
    print(f"\nMODEL ANALYSIS:")
    print(f"   Total nodes: {total_nodes}")
    print(f"   BatchNormalization nodes: {len(batchnorm_nodes)}")
    
    if len(batchnorm_nodes) == 0:
        print("   ✅ No BatchNorm nodes found - fusion successful!")
    else:
        print("   ⚠️  BatchNorm nodes still present:")
        for node_name in batchnorm_nodes:
            print(f"      - {node_name}")
    
    # Test inference with different batch sizes
    print(f"\nTESTING INFERENCE:")
    
    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()
    
    print(f"   Input: {input_info.name} {input_info.shape}")
    print(f"   Outputs: {[f'{out.name} {out.shape}' for out in output_info]}")
    
    # Test with batch size 1
    test_input_1 = np.random.randn(1, 1, 256, 256).astype(np.float32)
    try:
        outputs_1 = session.run(None, {input_info.name: test_input_1})
        print(f"   ✅ Batch size 1: Success - Output shapes: {[out.shape for out in outputs_1]}")
    except Exception as e:
        print(f"   ❌ Batch size 1: Failed - {e}")
        return
    
    # Test with batch size 4
    test_input_4 = np.random.randn(4, 1, 256, 256).astype(np.float32)
    try:
        outputs_4 = session.run(None, {input_info.name: test_input_4})
        print(f"   ✅ Batch size 4: Success - Output shapes: {[out.shape for out in outputs_4]}")
    except Exception as e:
        print(f"   ❌ Batch size 4: Failed - {e}")
        return
    
    # Verify consistency (same input should give same output)
    print(f"\nCONSISTENCY CHECK:")
    repeated_input = np.tile(test_input_1, (2, 1, 1, 1))  # [1,1,256,256] -> [2,1,256,256]
    outputs_repeated = session.run(None, {input_info.name: repeated_input})
    
    # Compare first output with single inference
    mu_diff = np.max(np.abs(outputs_1[0] - outputs_repeated[0][0:1]))
    logvar_diff = np.max(np.abs(outputs_1[1] - outputs_repeated[1][0:1]))
    
    print(f"   Max mu difference: {mu_diff:.2e}")
    print(f"   Max logvar difference: {logvar_diff:.2e}")
    
    if mu_diff < 1e-6 and logvar_diff < 1e-6:
        print("   ✅ Model is deterministic - BatchNorm working correctly!")
    else:
        print("   ⚠️  Differences detected - may indicate BatchNorm issues")
    
    print(f"\nSUMMARY:")
    if len(batchnorm_nodes) == 0:
        print("   ✅ BatchNorm fusion appears successful")
        print("   ✅ Model loads and runs correctly")
        print("   ✅ Dynamic batch size working")
    else:
        print("   ⚠️  BatchNorm nodes still present in graph")
        print("   ℹ️  This might be okay if ONNX runtime optimizes at inference time")
    
    print("=" * 50)

if __name__ == "__main__":
    # Usage
    onnx_model_path = sys.argv[1]  # Change this to your ONNX file path
    verify_onnx_fusion(onnx_model_path)
