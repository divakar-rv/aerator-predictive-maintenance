"""
Run this in the SAME Colab session where tflite_small was created (the
15.91 KB, 128-bin, 2-class vibration-only int8 model from Review-II).

Produces:
  1. model_data.h — the C header to place next to the .ino before flashing.
  2. Prints the exact input quantization scale/zero-point that MUST be
     copied into runInference() in aerator_predictive_maintenance.ino —
     the sketch currently has placeholder 0.0/0 values that will produce
     garbage predictions if left unset.

Usage (in Colab, after tflite_small exists in memory):
    exec(open('convert_model.py').read())
"""

import numpy as np

def convert_to_c_array(tflite_bytes, header_path, var_name="aerator_model"):
    with open(header_path, "w") as f:
        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, b in enumerate(tflite_bytes):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write(f"\n}};\nconst unsigned int {var_name}_len = {len(tflite_bytes)};\n")
    print(f"Wrote {header_path} ({len(tflite_bytes)} bytes -> {len(tflite_bytes)/1024:.2f} KB)")


def print_quantization_params(tflite_bytes):
    """Prints the exact values to paste into runInference() in the .ino."""
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zero = input_details[0]['quantization']
    print("\n=== COPY THESE INTO aerator_predictive_maintenance.ino ===")
    print(f"const float input_scale = {in_scale}f;")
    print(f"const int input_zero_point = {in_zero};")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output quantization (scale, zero_point): {output_details[0]['quantization']}")
    print("============================================================\n")

    print("Also verify the op list matches the MicroMutableOpResolver<N> calls")
    print("in setupTFLite() — list of ops actually used in this graph:")
    # Best-effort op listing via the flatbuffer; not exhaustive but a useful check
    try:
        import tensorflow.lite.python.schema_py_generated as schema_fb
        model = schema_fb.Model.GetRootAsModel(tflite_bytes, 0)
        subgraph = model.Subgraphs(0)
        op_codes = set()
        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode_index = op.OpcodeIndex()
            opcode = model.OperatorCodes(opcode_index)
            builtin = opcode.BuiltinCode()
            op_codes.add(builtin)
        print(op_codes, "(raw builtin op codes — cross-reference against")
        print("tensorflow/lite/builtin_ops.h to name them if unfamiliar)")
    except Exception as e:
        print(f"(Could not auto-list ops: {e}. Manually check by printing")
        print(" each layer of your Keras model — Conv1D, MaxPooling1D, Dense,")
        print(" Flatten map to TFLite's CONV_2D, MAX_POOL_2D, FULLY_CONNECTED,")
        print(" RESHAPE — TFLite converts 1D ops to their 2D equivalents")
        print(" internally, which is why the resolver uses *2D() calls.)")


# ---- Run both steps ----
if 'tflite_small' not in dir():
    print("ERROR: tflite_small not found in this session. Re-run the")
    print("quantization cell for the 15.91 KB model before this script.")
else:
    convert_to_c_array(tflite_small, "model_data.h", "aerator_model")
    print_quantization_params(tflite_small)
