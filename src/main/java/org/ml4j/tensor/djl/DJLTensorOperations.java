package org.ml4j.tensor.djl;

import ai.djl.ndarray.NDArray;
import org.jvmpy.symbolictensors.Operatable;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.tensor.TensorOperations;

public interface DJLTensorOperations extends TensorOperations<DJLTensorOperations>, Operatable<DJLTensorOperations, Size, DJLTensorOperations> {

	NDArray getNDArray();

	boolean isNativeGradient();
}
