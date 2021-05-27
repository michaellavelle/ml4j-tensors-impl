package org.ml4j.tensor.ml4j;

import org.jvmpy.symbolictensors.Operatable;
import org.ml4j.Matrix;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.tensor.Size;
import org.ml4j.tensor.TensorOperations;


public interface ML4JTensorOperations extends TensorOperations<ML4JTensorOperations>, Operatable<ML4JTensorOperations, Size, ML4JTensorOperations> {

	Matrix getMatrix();

	DirectedComponentsContext getDirectedComponentsContext();
		

}
