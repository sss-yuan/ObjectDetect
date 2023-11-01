import tensorflow as tf

# 第二步
converter = tf.lite.TFLiteConverter.from_saved_model("pack/") # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]


tflite_model = converter.convert()
# Save the model.
with open('tflite/objectdetect.tflite', 'wb') as f:
  f.write(tflite_model)
