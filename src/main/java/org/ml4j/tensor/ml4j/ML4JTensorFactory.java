package org.ml4j.tensor.ml4j;

import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import org.jvmpy.symbolictensors.Size;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.tensor.TensorFactory;
import org.ml4j.tensor.djl.DJLTensor;
import org.ml4j.tensor.djl.DJLTensorOperations;

import java.util.function.Supplier;

public class ML4JTensorFactory implements TensorFactory<ML4JTensor, ML4JTensorOperations> {

    public static final MatrixFactory DEFAULT_MATRIX_FACTORY = new JBlasRowMajorMatrixFactory();

    public static final DirectedComponentsContext DEFAULT_DIRECTED_COMPONENTS_CONTEXT = new DirectedComponentsContextImpl(DEFAULT_MATRIX_FACTORY, false);

    @Override
    public ML4JTensor create(Supplier<ML4JTensorOperations> supplier, Size size) {
        return new ML4JTensor(DEFAULT_DIRECTED_COMPONENTS_CONTEXT, supplier, size, false, false);
    }
}
